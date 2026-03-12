Here is a detailed, professional-grade GitHub README crafted specifically for your code. It breaks down the architecture, highlights your specific implementation details (like the per-patch normalization and un-shuffling logic), and frames your work perfectly for recruiters and other AI engineers.

---

# MAE-Vision: PyTorch Implementation of Masked Autoencoders 🎭

An optimized, from-scratch PyTorch implementation of the research paper **"Masked Autoencoders Are Scalable Vision Learners"** by Kaiming He et al.

This repository provides a complete, scalable pipeline for self-supervised visual representation learning. The MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. By masking a high proportion of the input image (e.g., 75%), we create a challenging self-supervisory task that forces the network to learn deep, holistic semantic representations.

## 🧠 Architecture Overview

This implementation strictly follows the asymmetric encoder-decoder design proposed in the original paper:

1. **Patchification & Masking:** The 224x224 input image is divided into a 14x14 grid of 16x16 pixel patches. We apply random masking to drop exactly 75% of these patches.


2. 
**ViT-Heavy Encoder:** The encoder (`ENC_LAYERS=12`, `ENC_DIM=768`) operates *only* on the 25% visible patches. It utilizes a `[CLS]` token and 2D sine-cosine positional embeddings. By ignoring mask tokens, it drastically reduces memory and compute costs.


3. 
**Unshuffling & Assembly:** The dense latent tokens are combined with shared, learned `[MASK]` tokens. Utilizing the saved indices (`ids_restore`), the 1D sequence is unshuffled back into its original spatial grid before entering the decoder.


4. 
**ViT-Light Decoder:** A lightweight Vision Transformer (`DEC_LAYERS=12`, `DEC_DIM=384`) processes the full sequence to reconstruct the missing pixels.


5. 
**Per-Patch Normalization Loss:** The MSE loss is computed exclusively on the masked patches. To improve representation quality, the target patches are locally normalized (using per-patch mean and variance) before computing the loss.



## ✨ Key Features in this Implementation

* **Mixed Precision Training:** Utilizes `torch.cuda.amp` (Autocast & GradScaler) for highly efficient VRAM usage and faster training loops.
* **Per-Patch Target Normalization:** Custom logic to normalize target pixels locally rather than globally, driving the model to focus on high-frequency details (textures and edges).
* **Distributed Data Parallel (DDP) Ready:** Code is structured to easily scale across multiple GPUs using `nn.DataParallel`.
* **Cosine Learning Rate Scheduler:** Implements a custom linear warmup followed by a cosine decay schedule for stable convergence.
* **Comprehensive Evaluation:** Built-in calculation of **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index) for quantitative reconstruction evaluation.

## 🛠️ Installation & Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/mae-vision.git
cd mae-vision
pip install torch torchvision numpy matplotlib scikit-image tqdm

```

## 📊 Dataset

This implementation is configured to train on **Tiny ImageNet** (or any standard ImageFolder dataset).
Images are dynamically resized to 224x224 and normalized using standard ImageNet statistics `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.

Update the `DATA_PATH` variable in the configuration block to point to your local dataset directory.

## 🚀 Training the Model

The model is highly configurable. Default hyperparameters:

* **Batch Size:** 64
* **Base LR:** 1.5e-4
* **Weight Decay:** 0.05
* **Epochs:** 20 (with 10 warmup epochs)

To train, simply run the training block or execute the script:

```python
python train.py

```

Checkpoints are automatically saved to `./checkpoints/mae_best.pth`.

## 📈 Evaluation & Visualizations

The repository includes utilities to visualize the 75% masked input, the model's raw reconstruction, and the ground truth. It seamlessly handles the un-normalization of both the global ImageNet stats and the local per-patch stats to generate clean images.

```python
# To generate PSNR/SSIM metrics and visualizations
python evaluate.py

```

*Reconstruction outputs and loss curves will be saved as `reconstructions.png` and `loss_curve.png` in your working directory.*

## 🤝 Contact & Author

Built with a passion for open-source AI and scalable computer vision systems.

**Nadeem Ahmad**

* Software & AI Engineer
* **LinkedIn:** [linkedin.com/in/nadeem-ahmad3](https://www.linkedin.com/in/nadeem-ahmad3/)
* **Email:** engrnadeem26@gmail.com

Feel free to reach out for collaborations, questions about the architecture, or discussions on self-supervised learning!

---

*If you find this repository helpful, please consider giving it a ⭐!*
