🩺 Retina Blood Vessel Segmentation
A deep learning project for binary semantic segmentation of blood vessels in retinal fundus images, built using PyTorch and a U-Net architecture with a pretrained ResNet-50 encoder.
---
📌 Project Overview
Retinal blood vessel segmentation is a critical step in diagnosing eye diseases such as diabetic retinopathy, glaucoma, and hypertensive retinopathy. This project automates the segmentation of blood vessels from fundus camera images using a state-of-the-art encoder-decoder model.
Property	Detail
Task	Binary Semantic Segmentation
Framework	PyTorch
Architecture	U-Net (ResNet-50 encoder)
Input Size	512 × 512 px
Loss Function	Dice Loss
Optimizer	Adam
Metric	IoU (Intersection over Union)
---
📁 Dataset
The dataset used is the Retina Blood Vessel Segmentation dataset, available on Kaggle. It contains paired retinal images and their corresponding binary vessel masks.
```
archive/
└── Data/
    ├── train/
    │   ├── image/   ← fundus images (.png)
    │   └── mask/    ← binary vessel masks (.png)
    └── test/
        ├── image/
        └── mask/
```
> **Note:** The dataset was accessed via Google Drive in the original notebook. Update the paths in the data-loading cells if running locally.
---
🔧 Setup & Installation
Prerequisites
Python 3.8+
CUDA-capable GPU (recommended)
Google Colab or local environment with GPU support
Install Dependencies
```bash
pip install torch torchvision
pip install albumentations
pip install segmentation-models-pytorch
pip install opencv-python pillow matplotlib pandas numpy tqdm scikit-learn seaborn
```
---
🏗️ Model Architecture
This project uses `segmentation-models-pytorch` to build a U-Net with a ResNet-50 encoder pretrained on ImageNet.
```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
```
Encoder: ResNet-50 (pretrained on ImageNet) — extracts rich feature maps at multiple scales.
Decoder: U-Net-style upsampling with skip connections from the encoder.
Output Head: Single-channel sigmoid output (binary mask prediction).
---
🔄 Data Pipeline
Augmentation (Training)
Augmentations are applied using the Albumentations library:
Augmentation	Detail
Resize	576 × 576
Random Crop	512 × 512
Horizontal/Vertical Flip	p = 0.5
Random Rotate 90°	p = 0.5
Shift-Scale-Rotate	Shift ±1%, Scale ±4%, Rotate ±15°
Preprocessing (All splits)
Pixel values normalized to `[0, 1]` by dividing by 255.
Masks binarized: pixels ≥ 127 → `1` (vessel), pixels < 127 → `0` (background).
Images converted to PyTorch tensors with shape `(C, H, W)`.
---
🏋️ Training
```python
EPOCHS     = 50
BATCH_SIZE = 8
LR         = 0.001
```
Component	Choice
Loss Function	`smp.losses.DiceLoss(mode="binary")`
Optimizer	`torch.optim.Adam`
LR Scheduler	`StepLR(step_size=100, gamma=0.1)`
Early Stopping	Patience = 5 epochs (on val loss)
Checkpointing	`checkpoints/best.pth` (best val loss)
Run the notebook cell by cell, or adapt the training loop for a `.py` script.
---
📊 Evaluation Metrics
The model is evaluated on the held-out test set using:
Metric	Description
IoU	Intersection over Union (primary segmentation metric)
Accuracy	Pixel-level accuracy
F1-Score	Harmonic mean of precision and recall
Precision	TP / (TP + FP)
Recall	TP / (TP + FN)
Metrics are computed using `smp.metrics` at a threshold of `0.5`.
---
📈 Results
Training and validation curves for Loss and IoU Score are plotted after training.
The final cell visualizes qualitative predictions by showing:
Input — original fundus image
Prediction — model-predicted vessel mask
Ground Truth — annotated mask
---
📂 Project Structure
```
retina-segmentation/
│
├── Retina_segmentation_with_output.ipynb   ← Main notebook
├── checkpoints/
│   ├── best.pth                            ← Best model weights (saved during training)
│   └── last.pth                            ← Last epoch weights
└── README.md
```
---
🚀 Inference
To run inference on new images using the saved best model:
```python
import torch
from PIL import Image
import numpy as np

model.load_state_dict(torch.load("checkpoints/best.pth"))
model.to(device)
model.eval()

# Preprocess image (resize to 512×512, normalize, convert to tensor)
# ...

with torch.no_grad():
    pred = model(img_tensor.unsqueeze(0))
    pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    pred_mask = (pred > 0.5).astype(np.uint8)
```
---
🧰 Libraries Used
Library	Purpose
`PyTorch`	Deep learning framework
`segmentation-models-pytorch`	Pretrained U-Net architectures
`Albumentations`	Image augmentation
`OpenCV`	Image I/O and color conversion
`Matplotlib`	Visualization
`scikit-learn`	Train/test split, metrics
`tqdm`	Progress bars
`Pandas / NumPy`	Data manipulation
---
📝 Notes
The notebook was originally developed and run on Google Colab with GPU runtime.
Dataset paths reference Google Drive mounts — update them if running locally.
`encoder_weights="imagenet"` enables transfer learning; the encoder is fine-tuned during training.
---
📜 License
This project is for academic and research purposes.
