"""
Real-Time Facial Emotion Recognition — GUI
ResNet50 + STN + CBAM (69.0% on FER2013)

Requirements:
    pip install torch torchvision opencv-python pillow tkinter
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image, ImageTk
import threading
import os

# ══════════════════════════════════════════════════════════════════════════════
# Model Architecture
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
EMOTION_COLORS = {
    'angry'   : '#EF4444',
    'disgust' : '#84CC16',
    'fear'    : '#8B5CF6',
    'happy'   : '#F59E0B',
    'neutral' : '#94A3B8',
    'sad'     : '#3B82F6',
    'surprise': '#EC4899',
}
EMOTION_EMOJIS = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲',
}
BG       = "#0D1B2A"
BG_CARD  = "#1A2E44"
TEAL     = "#0D9488"
WHITE    = "#FFFFFF"
GRAY     = "#94A3B8"


# ══════════════════════════════════════════════════════════════════════════════
# Main GUI Application
# ══════════════════════════════════════════════════════════════════════════════
class EmotionApp:
    def __init__(self, root):
        self.root       = root
        self.root.title("Real-Time Facial Emotion Recognition — ResNet50 + STN + CBAM")
        self.root.geometry("1100x700")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.model        = None
        self.class_names  = []
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform    = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.cap          = None
        self.running      = False
        self.frame_count  = 0
        self.last_emotion = ""
        self.last_conf    = 0.0
        self.last_probs   = []

        self._build_ui()

    # ── UI Layout ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar
        top = tk.Frame(self.root, bg=TEAL, height=50)
        top.pack(fill="x")
        tk.Label(top, text="🎭  Real-Time Facial Emotion Recognition",
                 font=("Segoe UI", 14, "bold"), bg=TEAL, fg=WHITE).pack(side="left", padx=15, pady=10)
        tk.Label(top, text=f"Device: {self.device}",
                 font=("Segoe UI", 10), bg=TEAL, fg=WHITE).pack(side="right", padx=15)

        # ── Main content
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=10, pady=10)

        # Left — video feed
        left = tk.Frame(content, bg=BG_CARD, relief="flat", bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        tk.Label(left, text="📷  Live Camera Feed",
                 font=("Segoe UI", 11, "bold"), bg=BG_CARD, fg=WHITE).pack(pady=(10, 5))

        self.video_label = tk.Label(left, bg="black")
        self.video_label.pack(padx=10, pady=5, fill="both", expand=True)

        # Image mode label (shown when image is loaded)
        self.img_label = tk.Label(left, bg="black")

        # ── Control buttons
        btn_frame = tk.Frame(left, bg=BG_CARD)
        btn_frame.pack(pady=10)

        self._btn(btn_frame, "📂  Load Model", TEAL,       self.load_model).pack(side="left", padx=5)
        self._btn(btn_frame, "▶  Start Camera", "#22C55E", self.start_camera).pack(side="left", padx=5)
        self._btn(btn_frame, "⏹  Stop Camera",  "#EF4444", self.stop_camera).pack(side="left", padx=5)
        self._btn(btn_frame, "🖼  Load Image",   "#F59E0B", self.load_image).pack(side="left", padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="⚠️  Please load model first")
        tk.Label(left, textvariable=self.status_var,
                 font=("Segoe UI", 9), bg=BG_CARD, fg=GRAY).pack(pady=(0, 8))

        # Right panel
        right = tk.Frame(content, bg=BG, width=300)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        # Emotion display
        emo_card = tk.Frame(right, bg=BG_CARD, relief="flat")
        emo_card.pack(fill="x", pady=(0, 8))

        tk.Label(emo_card, text="Detected Emotion",
                 font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=GRAY).pack(pady=(10, 0))

        self.emoji_label = tk.Label(emo_card, text="🎭",
                                    font=("Segoe UI", 48), bg=BG_CARD, fg=WHITE)
        self.emoji_label.pack()

        self.emotion_label = tk.Label(emo_card, text="—",
                                      font=("Segoe UI", 20, "bold"), bg=BG_CARD, fg=WHITE)
        self.emotion_label.pack()

        self.conf_label = tk.Label(emo_card, text="Confidence: —",
                                   font=("Segoe UI", 11), bg=BG_CARD, fg=GRAY)
        self.conf_label.pack(pady=(0, 12))

        # Probability bars
        bars_card = tk.Frame(right, bg=BG_CARD)
        bars_card.pack(fill="both", expand=True)

        tk.Label(bars_card, text="Class Probabilities",
                 font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=GRAY).pack(pady=(10, 6))

        self.bar_vars   = {}
        self.bar_labels = {}
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        for cls in emotions:
            row = tk.Frame(bars_card, bg=BG_CARD)
            row.pack(fill="x", padx=10, pady=2)

            color = EMOTION_COLORS[cls]
            emoji = EMOTION_EMOJIS[cls]

            tk.Label(row, text=f"{emoji} {cls.capitalize()}",
                     font=("Segoe UI", 9), bg=BG_CARD, fg=WHITE,
                     width=12, anchor="w").pack(side="left")

            bar_bg = tk.Frame(row, bg="#1E3A5F", height=14, relief="flat")
            bar_bg.pack(side="left", fill="x", expand=True, padx=(4, 6))
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=color, height=14)
            bar_fill.place(x=0, y=0, relwidth=0.0, height=14)
            self.bar_vars[cls] = bar_fill

            pct_lbl = tk.Label(row, text="0.0%",
                               font=("Segoe UI", 9), bg=BG_CARD, fg=GRAY, width=5)
            pct_lbl.pack(side="left")
            self.bar_labels[cls] = pct_lbl

        # Model info
        self.model_info = tk.Label(right, text="Model: Not loaded",
                                   font=("Segoe UI", 8), bg=BG, fg=GRAY)
        self.model_info.pack(pady=6)

    def _btn(self, parent, text, color, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg=WHITE, font=("Segoe UI", 9, "bold"),
                         relief="flat", padx=10, pady=6, cursor="hand2",
                         activebackground=color, activeforeground=WHITE)

    # ── Load Model ────────────────────────────────────────────────────────
    def load_model(self):
        path = filedialog.askopenfilename(
            title="Select emotion_model_webcam.pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            self.status_var.set("⏳  Loading model...")
            self.root.update()

            bundle           = torch.load(path, map_location=self.device)
            self.class_names = bundle["class_names"]
            num_classes      = bundle["num_classes"]

            self.model = ResNet50WithSTNAndCBAM(num_classes).to(self.device)
            self.model.load_state_dict(bundle["model_state_dict"])
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=bundle["normalize_mean"],
                    std=bundle["normalize_std"]
                )
            ])

            acc = bundle.get("test_accuracy", 0)
            self.model_info.config(text=f"Model: ResNet50+STN+CBAM  |  Accuracy: {acc:.2f}%  |  Device: {self.device}")
            self.status_var.set(f"✅  Model loaded — {acc:.2f}% test accuracy")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self.status_var.set("❌  Model load failed")

    # ── Start / Stop Camera ───────────────────────────────────────────────
    def start_camera(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please load the model first.")
            return
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return

        self.running     = True
        self.frame_count = 0
        self.status_var.set("🎥  Camera running — Press Stop to quit")
        self.img_label.pack_forget()
        self.video_label.pack(padx=10, pady=5, fill="both", expand=True)

        thread = threading.Thread(target=self._camera_loop, daemon=True)
        thread.start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image="")
        self.status_var.set("⏹  Camera stopped")

    def _camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            frame = self._process_frame(frame)

            # Convert for tkinter
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            img   = img.resize((640, 460), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

    # ── Load Image ────────────────────────────────────────────────────────
    def load_image(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please load the model first.")
            return

        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        if not path:
            return

        self.stop_camera()

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Could not read image.")
            return

        frame = self._process_frame(frame)

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        img   = img.resize((640, 460), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.pack_forget()
        self.img_label.pack(padx=10, pady=5, fill="both", expand=True)
        self.img_label.imgtk = imgtk
        self.img_label.config(image=imgtk)
        self.status_var.set(f"🖼  Image loaded: {os.path.basename(path)}")

    # ── Process Frame (inference + drawing) ───────────────────────────────
    def _process_frame(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if self.frame_count % 3 == 0 and len(faces) > 0 and self.model is not None:
            x, y, w, h = faces[0]
            face_crop  = frame[y:y+h, x:x+w]
            face_pil   = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            tensor     = self.transform(face_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                probs           = torch.softmax(self.model(tensor), dim=1)[0]
                pred_idx        = torch.argmax(probs).item()
                self.last_emotion = self.class_names[pred_idx]
                self.last_conf    = probs[pred_idx].item() * 100
                self.last_probs   = probs.cpu().tolist()

            self.root.after(0, self._update_right_panel)

        # Draw bounding box
        color_bgr = self._hex_to_bgr(EMOTION_COLORS.get(self.last_emotion, "#FFFFFF"))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 2)
            if self.last_emotion:
                label = f"{self.last_emotion.upper()} {EMOTION_EMOJIS.get(self.last_emotion,'')}  {self.last_conf:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(frame, (x, y-th-14), (x+tw+8, y), color_bgr, -1)
                cv2.putText(frame, label, (x+4, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        return frame

    def _update_right_panel(self):
        if not self.last_emotion:
            return

        # Update emotion display
        emoji = EMOTION_EMOJIS.get(self.last_emotion, "🎭")
        color = EMOTION_COLORS.get(self.last_emotion, WHITE)
        self.emoji_label.config(text=emoji)
        self.emotion_label.config(text=self.last_emotion.capitalize(), fg=color)
        self.conf_label.config(text=f"Confidence: {self.last_conf:.1f}%")

        # Update bars
        if self.last_probs and self.class_names:
            for i, cls in enumerate(self.class_names):
                if cls in self.bar_vars:
                    p = self.last_probs[i]
                    self.bar_vars[cls].place(relwidth=p)
                    self.bar_labels[cls].config(text=f"{p*100:.1f}%")

    @staticmethod
    def _hex_to_bgr(hex_color):
        h   = hex_color.lstrip('#')
        r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return (b, g, r)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = EmotionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
