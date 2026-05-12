# Real-Time Facial Emotion Recognition
### ResNet50 + Spatial Transformer Network + CBAM on FER2013
**MSA University — Faculty of Computer Science**
**Course: Pattern Recognition (CS487)**

---

## 👥 Team Members
| ID     | Name            |
|--------|-----------------|
| 247825 | Ahmed Bassiouny |

---

## 📊 Results Summary

| Model                              | Accuracy | F1 Weighted | F1 Macro |
|------------------------------------|----------|-------------|----------|
| 1. Baseline ResNet50               | 63.83%   | 0.6374      | 0.5963   |
| 2. ResNet50 + Label Smoothing      | 65.29%   | 0.6525      | 0.6077   |
| 3. ResNet50 + CBAM                 | 65.93%   | 0.6588      | 0.6063   |
| 4a. ResNet50 + STN + CBAM (30ep)   | 66.11%   | 0.6633      | 0.6180   |
| **4b. ResNet50 + STN + CBAM (50ep)**| **69.0%**| **0.6898**  | **0.6707**|
| 5. EmoNeXt (ConvNeXt-Small)        | 64.03%   | 0.6322      | 0.5882   |

---

## 📁 Folder Structure

```
FER_Project/
│
├── Dataset/
│   └── dataset_info.txt          ← FER2013 download instructions
│
├── Models/
│   ├── emotion_model_webcam.pth  ← Full bundle (weights + class names + metadata)
│   └── model4_stn_cbam_best.pth  ← Best model weights only
│
├── Code/
│   ├── train_models_1_to_4.ipynb ← Training notebooks (run on Kaggle/Colab)
│   ├── train_model4_50epochs.ipynb
│   ├── webcam.py                 ← Real-time webcam inference (no GUI)
│   └── gui.py                    ← Full desktop GUI application
│
├── Evaluation/
│   ├── comparison_chart_final.png
│   ├── confusion_matrix_best_model.png
│   ├── confusion_matrix_resnet50_cbam.png
│   └── confusion_matrix_baseline.png
│
├── Presentation/
│   └── FER_Presentation.pptx
│
└── Paper/
    └── IMSA_Paper.pdf
```

---

## ⚙️ Setup & Installation

```bash
pip install torch torchvision opencv-python pillow scikit-learn matplotlib seaborn
```

---

## 🚀 How to Run

### Option 1 — GUI (Recommended)
```bash
python Code/gui.py
```
1. Click **Load Model** → select `Models/emotion_model_webcam.pth`
2. Click **Start Camera** for live webcam detection
3. Or click **Load Image** to analyze a single photo
4. Press **Stop Camera** or close the window to quit

### Option 2 — Webcam Script (No GUI)
```bash
# Edit webcam.py line 45 — set WEBCAM_MODEL_PATH to your model path
python Code/webcam.py
# Press Q to quit
```

### Option 3 — Training (Kaggle / Google Colab)
1. Upload `Code/train_models_1_to_4.ipynb` to Kaggle or Colab
2. Set dataset path to your FER2013 folder
3. Run all cells

---

## 📦 Dataset

**FER2013** — available on Kaggle:
```
https://www.kaggle.com/datasets/damnithurts/fer2013-dataset-images
```

- 35,887 grayscale images at 48×48 px
- 7 classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Train: 28,709 | Test: 7,178

---

## 🏗️ Model Architecture

```
Input Image (224×224×3)
        ↓
   STN (Spatial Transformer Network)     ← geometric alignment
        ↓
   ResNet50 Backbone (pretrained ImageNet, layer4 fine-tuned)
        ↓
   CBAM (Channel Attention + Spatial Attention on 7×7×2048)
        ↓
   Global Average Pool → FC (2048 → 7)
        ↓
   Softmax → Predicted Emotion
```

---

## 🎯 Key Design Choices

| Technique | Purpose |
|-----------|---------|
| Transfer Learning (ImageNet) | Strong feature initialization |
| Label Smoothing (ε=0.1) | Handles noisy crowdsourced labels |
| Class Weighting | Handles severe class imbalance |
| MixUp Augmentation | Reduces overfitting |
| STN (frozen 10ep then unfrozen) | Geometric face alignment |
| CBAM Attention | Focus on discriminative facial regions |
| 50 epochs, no early stopping | Full convergence of STN+CBAM |
| Haar Cascade (inference) | Real-time face detection |

---

## 📈 Training Details

- **Optimizer:** Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR
- **Batch size:** 32
- **Epochs:** 50 (no early stopping for best model)
- **STN:** frozen epochs 1–10, unfrozen with lr=5e-5 + gradient clipping (max_norm=1.0)
- **Platform:** Kaggle / Google Colab (T4 GPU)
