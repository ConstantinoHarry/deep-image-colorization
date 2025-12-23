## Deep Image Colorization (ResNet-UNet GAN)
Hong Kong Baptist University — COMP3057 AI/ML (Course Project), Dec 2025

- Built a conditional GAN for automatic grayscale → color image translation in Lab color space, using a `ResNet-18`-backed `U-Net` generator and `PatchGAN` discriminator.
- Designed a hybrid objective combining L1 reconstruction, adversarial loss, and perceptual `LPIPS` (VGG) to prioritize human-perceived realism over purely pixel metrics.
- Implemented retrieval-augmented color guidance with `FAISS` nearest-neighbor hints to mitigate multimodal color ambiguity and reduce desaturation artifacts.
- Engineered a stable training pipeline: warmup (L1-only) for 20 epochs, TTUR-like optimizers, spectral normalization in D, gradient clipping, and StepLR scheduling; produced checkpoints and qualitative galleries.
- Evaluated on a curated subset (~1,500 images) of the Kaggle Image Colorization Dataset; achieved average PSNR ≈ 22.53 dB and SSIM ≈ 0.9162 on 100 test images.
- Stack: Python, PyTorch (`torchvision`, `lpips`), `faiss-cpu`, `scikit-image`, `matplotlib`; trained on Google Colab Tesla T4 GPU.

Selected Deliverables: Notebook with reproducible pipeline (`Image_Colorization_ResNet_UNet_GAN.ipynb`), model checkpoints, evaluation metrics, and result galleries.