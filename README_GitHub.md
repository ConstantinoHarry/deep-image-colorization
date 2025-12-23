# Deep Image Colorization (ResNet-UNet GAN)

Semantic-aware grayscale → color translation using a Pix2Pix-style cGAN. The generator combines a pre-trained ResNet-18 backbone with a U-Net decoder; a PatchGAN discriminator enforces local realism. A hybrid objective (L1 + GAN + LPIPS perceptual loss) and optional FAISS retrieval hints improve vibrancy and stability. Colab-ready.

<img src="Images/ProjectHeader.png" alt="Grayscale vs. Colorized example" width="720" />

## Features
- ResNet-18 + U-Net generator; PatchGAN discriminator
- Hybrid loss: L1 + adversarial + LPIPS (VGG) perceptual
- Optional FAISS retrieval hints for color priors and stability
- Lab color space conditioning (predict `ab` from grayscale `L`)
- Colab-friendly notebook with reproducible setup, training, evaluation, and galleries

## Quickstart

### Option A: Run on Google Colab (Recommended)
1. Open `Image_Colorization_ResNet_UNet_GAN.ipynb` in Colab.
2. Run the Setup cell (installs packages and sets device).
3. Download the dataset via Kaggle API (instructions in notebook).
4. Run training cells, then evaluation to see metrics and galleries.

### Option B: Run Locally (macOS/Linux/Windows)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install torch torchvision torchaudio scikit-image matplotlib tqdm lpips faiss-cpu
```

> If using CUDA locally, install the matching `torch` build from https://pytorch.org/get-started/locally/

## Dataset
- Source: Kaggle “Image Colorization Dataset” (~5,000 natural/landscape images)
- Subset: ~1,500 curated images for efficient training
- Train/Val/Test: ~80/10/10 split
- Preprocessing: RGB→Lab, normalize `L` to [-1,1], scale `ab` to [-1,1], resize to 256×256

## Model Overview
- Generator (`ResNetUNet`): ResNet-18 encoder (grayscale-adapted), U-Net decoder with skip connections; outputs 2-channel `ab` with `tanh`
- Discriminator: PatchGAN classifying local patches for sharpness
- Retrieval (Optional): FAISS `IndexFlatL2` over training embeddings to provide color priors
- Losses: L1 reconstruction + adversarial (BCE/Hinge) + LPIPS perceptual (λ_perc=10)
- Training: Warmup (L1-only) for 20 epochs; full cGAN training to ~120 epochs; TTUR-like optimizers, spectral norm in D, gradient clipping, StepLR

## Results
- PSNR ≈ 22.53 dB, SSIM ≈ 0.9162 on 100 test images
- Qualitative galleries and comparisons are included in the notebook
- Strong semantic colorization on common scenes; some ambiguity on rare objects

## Single-Image Inference (Notebook Snippet)
```python
# Given: generator loaded on device; L in [-1,1] as (1,1,256,256)
with torch.no_grad():
    ab_hat = generator(L.to(device))  # (1,2,256,256), tanh in [-1,1]
from skimage.color import lab2rgb
import numpy as np
L_np  = ((L.cpu().numpy() + 1.0) * 50.0)[0,0]                 # [0,100]
ab_np = (ab_hat.cpu().numpy() * 128.0)[0].transpose(1,2,0)   # [-128,127]
lab   = np.dstack([L_np, ab_np])
rgb   = lab2rgb(lab)                                          # [0,1]
```

## Troubleshooting
- ResNet weights fail to download: ensure internet or set `pretrained=False` temporarily
- LPIPS errors: tensors must be NCHW in `[-1,1]` and on the same device
- FAISS unavailable: use `faiss-cpu`, or disable retrieval hints
- Colors look washed/sepiatone: verify Lab scaling consistency in train/eval
- CUDA OOM: reduce batch size or disable heavy augmentations

## Repository Structure
```
├── README_GitHub.md
├── README.md
├── Image_Colorization_ResNet_UNet_GAN.ipynb
├── Images/
├── colorizer_*.pth            # Saved models (during training)
└── data/                      # Dataset folder (populated during run)
```

## Tech Stack
Python, PyTorch (`torchvision`, `lpips`), `faiss-cpu`, `scikit-image`, `matplotlib`; trained on Google Colab Tesla T4 GPU.

## Citation & References
- Isola et al., 2017 — Pix2Pix
- Ronneberger et al., 2015 — U-Net
- Zhang et al., 2016 — Colorful Image Colorization
- Zhang et al., 2018 — LPIPS perceptual metric

## License
Add a license of your choice (e.g., MIT/Apache-2.0) and update this section.

## Suggested GitHub Topics
image-colorization, gan, unet, resnet, pytorch, lpips, faiss, computer-vision, colab, deep-learning
