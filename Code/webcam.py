import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image

# ── Architecture (must match training) ────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg  = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg     = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel = ChannelAttention(in_channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))

class STN(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8,  kernel_size=7), nn.MaxPool2d(2, stride=2), nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5), nn.MaxPool2d(2, stride=2), nn.ReLU(True),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 256), nn.ReLU(True),
            nn.Linear(256, 6)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs    = self.localization(x).view(x.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid  = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

class ResNet50WithSTNAndCBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stn      = STN()
        base          = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.cbam     = CBAM(2048)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stn(x)
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        return self.fc(x.flatten(1))

# ── Load model from webcam bundle ─────────────────────────────────────────
WEBCAM_MODEL_PATH = r"D:\Downloads\FER_Project\FER_Project\Models\emotion_model_webcam.pth"

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle  = torch.load(WEBCAM_MODEL_PATH, map_location=device)

class_names = bundle["class_names"]
num_classes = bundle["num_classes"]

model = ResNet50WithSTNAndCBAM(num_classes).to(device)
model.load_state_dict(bundle["model_state_dict"])
model.eval()

print(f"✅ Model loaded | Accuracy: {bundle['test_accuracy']:.2f}%")
print(f"   Classes: {class_names}")
print(f"   Device : {device}")

# ── Transform ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=bundle["normalize_mean"],
        std=bundle["normalize_std"]
    )
])

# ── Emoji map ─────────────────────────────────────────────────────────────
EMOJI = {
    'angry'   : '😠',
    'disgust' : '🤢',
    'fear'    : '😨',
    'happy'   : '😊',
    'neutral' : '😐',
    'sad'     : '😢',
    'surprise': '😲',
}

# ── Colors per emotion (BGR) ───────────────────────────────────────────────
COLORS = {
    'angry'   : (0,   0,   255),
    'disgust' : (0,   128, 0  ),
    'fear'    : (128, 0,   128),
    'happy'   : (0,   255, 255),
    'neutral' : (200, 200, 200),
    'sad'     : (255, 0,   0  ),
    'surprise': (0,   165, 255),
}

# ── Face detector ─────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Webcam loop ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Press Q to quit")

frame_count    = 0
last_emotion   = ""
last_confidence= 0.0
last_probs     = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # ── Run inference every 3 frames ──────────────────────────────────────
    if frame_count % 3 == 0 and len(faces) > 0:
        x, y, w, h   = faces[0]                          # use largest face
        face_crop    = frame[y:y+h, x:x+w]
        face_pil     = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        img_tensor   = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            probs          = torch.softmax(model(img_tensor), dim=1)[0]
            pred_idx       = torch.argmax(probs).item()
            last_emotion   = class_names[pred_idx]
            last_confidence= probs[pred_idx].item() * 100
            last_probs     = probs.cpu()

    # ── Draw face boxes ───────────────────────────────────────────────────
    for (x, y, w, h) in faces:
        color = COLORS.get(last_emotion, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        if last_emotion:
            label = f"{last_emotion.upper()} {EMOJI.get(last_emotion,'')}  {last_confidence:.1f}%"
            # background pill behind text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y-th-14), (x+tw+8, y), color, -1)
            cv2.putText(frame, label, (x+4, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # ── Sidebar probability bars ──────────────────────────────────────────
    if len(last_probs) > 0:
        bar_x     = 10
        bar_y_start = 20
        bar_w_max = 180
        bar_h     = 18
        gap       = 28

        for i, (cls, prob) in enumerate(zip(class_names, last_probs)):
            p      = prob.item()
            filled = int(p * bar_w_max)
            by     = bar_y_start + i * gap
            color  = COLORS.get(cls, (200, 200, 200))

            # background bar
            cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w_max, by + bar_h),
                          (50, 50, 50), -1)
            # filled bar
            cv2.rectangle(frame, (bar_x, by), (bar_x + filled, by + bar_h),
                          color, -1)
            # label
            text = f"{cls}: {p*100:.1f}%"
            cv2.putText(frame, text, (bar_x + bar_w_max + 6, by + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Emotion Detection — ResNet50 + STN + CBAM (69.0%)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()