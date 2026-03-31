# SeisDiff-FNO: Seismic Data Enhancement Using Fourier Neural Operators within a Conditional Diffusion Framework

<div align="center">
  <a href="https://github.com/traversa942" target="_blank">Alessandro Traversa<sup>1</sup></a> &emsp;
  <a href="https://cpg.kfupm.edu.sa/bio/dr-umair-bin-waheed/" target="_blank">Umair Bin Waheed<sup>1</sup></a> &emsp;
  <a href="" target="_blank">Abdulmohsen AlAli<sup>2</sup></a> &emsp;
  <a href="" target="_blank">Tariq A. Alkhalifah<sup>3</sup></a>
</div>

<div align="center">
  <sup>1</sup> College of Petroleum Engineering & Geosciences, King Fahd University of Petroleum & Minerals <br>
  <sup>2</sup> EXPEC Advanced Research Center, Saudi Aramco <br>
  <sup>3</sup> Earth Science and Engineering Program, King Abdullah University of Science and Technology
</div>

## Abstract

Seismic data quality is often degraded by noise contamination, limited acquisition bandwidth, and sparse sampling, reducing subsurface imaging reliability. Recent deep learning advances, including U-Net architectures and diffusion models, have improved denoising, super-resolution, and missing-trace reconstruction, yet remain constrained by local receptive fields or high computational costs. To address these limitations, this work proposes SeisDiff-FNO, a conditional diffusion framework integrating Fourier Neural Operator (FNO) layers into the U-Net backbone to enhance long-range spatial modeling while maintaining computational efficiency. The model was trained using synthetic datasets with paired low/high-resolution seismic sections derived from 3D reflectivity models with varying geological complexity and noise levels. Comprehensive experiments were conducted across three enhancement tasks: super-resolution with factor-of-two upsampling, denoising under multiple noise regimes, and reconstruction under random missing-trace patterns. Results demonstrate that SeisDiff-FNO consistently outperforms U-Net, and standard diffusion models with and without attention mechanism in spatial and spectral domains, recovering broadband frequency-wavenumber information, improving reflector continuity, and reducing residual artifacts. Inference times were substantially reduced relative to attention-augmented baselines. The model exhibited strong generalization on out-of-distribution field-like seismic data.

## Reference

    Traversa, Alessandro, Umair Bin Waheed, Abdulmohsen AlAli, and Tariq A. Alkhalifah.
    "SeisDiff-FNO: Seismic Data Enhancement Using Fourier Neural Operators within a
    Conditional Diffusion Framework." IEEE Transactions on Geoscience and Remote Sensing,
    VOL. 18, NO. 9, 2026.

BibTeX

    @article{traversa2026seisdiff,
      title={SeisDiff-FNO: Seismic Data Enhancement Using Fourier Neural Operators within a Conditional Diffusion Framework},
      author={Traversa, Alessandro and Waheed, Umair Bin and AlAli, Abdulmohsen and Alkhalifah, Tariq A.},
      journal={IEEE Transactions on Geoscience and Remote Sensing},
      volume={18},
      number={9},
      year={2026},
      publisher={IEEE}
    }

## Data Preparation

The training data preparation workflow can be divided into three steps:

1. **Downloading and generating the synthetic seismic dataset:** We use the synthetic seismic modeling framework developed by [Li et al. (2021)](https://github.com/JintaoLee-Roger/SeismicSuperResolution) to generate paired low-resolution (LR) and high-resolution (HR) seismic sections. The original dataset consists of 3,200 paired 2D slices derived from 800 synthetic 3D reflectivity volumes with varying fold intensity, fault density, reflector geometry, peak frequencies, and noise levels.

2. **Geophysically informed data augmentation:** To expand the dataset from 3,200 to 10,000 samples, we apply a series of physically consistent transformations to the LR sections, including:
   - Horizontal flipping
   - Amplitude scaling within [0.8, 1.2]
   - Trace dropout (up to 40% of LR traces)
   - Coherent low-frequency noise (simulating residual ground roll)
   - Spatially correlated Gaussian noise
   - Trace gain variations
   - Localized amplitude spikes
   - Mild phase perturbations

3. **Task-specific conditioning:** For each enhancement task (super-resolution, denoising, trace reconstruction), the LR section is used as the conditioning signal for the diffusion model. Missing-trace masks are applied randomly during training to cover the reconstruction task.

## Model Architecture

SeisDiff-FNO is built upon a conditional DDPM framework employing a modified U-Net architecture that strategically integrates CNN and FNO layers within ResNet blocks. The key components are:

- **Hybrid ResNet Blocks:** Each block combines convolutional layers for local feature extraction with FNO layers for global spectral modeling via spectral convolution:

      (K * u) = F⁻¹(R · F(u))

  where R represents learnable spectral weights, F denotes the Fourier transform, and u is the input feature map.

- **Encoder-Decoder Structure:** Four resolution levels with three ResNet blocks each. Initial convolution projects input from 1 to 12 channels; final convolution reduces output to 1 channel. Feature dimensions progress as [48, 96, 192, 384].

- **Conditioning Mechanism:** The low-resolution guidance signal is spatially aligned via bilinear interpolation at each resolution level and integrated through element-wise addition, influencing both local texture extraction and global structural modeling.

- **Timestep Conditioning:** Sinusoidal positional embeddings processed through an MLP are injected into every ResNet block via affine transformations with learned scale and shift parameters.

- **Diffusion Framework:** Forward process follows a cosine noise schedule over T = 1,000 timesteps. The reverse process is based on the Improved DDPM formulation, jointly predicting both noise and log-variance at each step.

- **Loss Function:** Combines Min-SNR-weighted noise prediction loss with the variational lower bound (VLB) term from Improved DDPM:

      L = w(t) · MSE(ε̂_θ(x_t, t, c), ε) + λ · L_vlb(t)

  with γ = 2 (Min-SNR threshold) and λ = 0.001 (VLB balance parameter).

## Training Process

The training process can be divided into 4 steps:

1. **Training data:** 10,000 augmented paired LR-HR seismic sections, split 90% training / 10% validation.

2. **Conditioning:** The LR section (128×128) is passed as the conditioning signal c to guide the reverse diffusion process toward the HR target (256×256).

3. **Establish the Hyperparameters:**

   - **Number of Epochs:** 150 (with early stopping, patience = 24)
   - **Learning Rate:** 1e-4
   - **Batch Size:** 1 (due to memory constraints)
   - **Optimizer:** Adam
   - **Timesteps:** 1,000
   - **Noise Schedule:** Cosine (s = 0.008)
   - **GPU:** NVIDIA RTX A4500

4. **Train the model with the input.**

## How to Run the Training Code?

1. **Step 1: Obtain the dataset:** Download and generate the synthetic seismic dataset following [Li et al. (2021)](https://github.com/JintaoLee-Roger/SeismicSuperResolution). The original 3,200 LR-HR pairs serve as the base.

2. **Step 2: Run data augmentation:** Apply the geophysically informed augmentation pipeline to expand the dataset to 10,000 samples. Save the augmented pairs as `.npy` files.

3. **Step 3: Run the training code:** The training script includes visualization steps to monitor the diffusion process and reconstruction quality throughout training.

4. **Step 4: Download pretrained model weights:** Due to file size limitations, pretrained model weights are available on Google Drive (link to be added upon publication). Place the downloaded weights in the `model/` folder.

## Experimental Results

SeisDiff-FNO is benchmarked against three baselines: supervised U-Net (Li et al., 2021), standard conditional diffusion (SeisDiff), and SeisFusion (Wang et al., 2024). All diffusion models maintain approximately equal capacity (~490M parameters).

**Super-Resolution (2× upsampling, complex geology):**

| Metric | U-Net | SeisFusion | SeisDiff | SeisDiff-FNO |
|---|---|---|---|---|
| MSE (↓) | 0.073 | 0.026 | 0.040 | **0.012** |
| SNR (dB) (↑) | 11.338 | 15.759 | 13.914 | **19.262** |
| PSNR (dB) (↑) | 27.628 | 32.112 | 30.466 | **35.646** |
| SSIM (↑) | 0.838 | 0.944 | 0.913 | **0.975** |
| Inference Time (s) (↓) | **0.207** | 154.952 | 211.653 | 106.407 |

**Denoising (medium noise):**

| Metric | SeisFusion | SeisDiff | SeisDiff-FNO |
|---|---|---|---|
| MSE (↓) | 0.067 | 0.091 | **0.044** |
| SNR (dB) (↑) | 11.753 | 10.430 | **13.539** |
| SSIM (↑) | 0.906 | 0.832 | **0.947** |

**Trace Reconstruction (40% missing):**

| Metric | SeisFusion | SeisDiff | SeisDiff-FNO |
|---|---|---|---|
| MSE (↓) | 0.072 | 0.117 | **0.062** |
| SNR (dB) (↑) | 11.413 | 9.333 | **12.096** |
| SSIM (↑) | 0.895 | 0.825 | **0.923** |

## Development

The development team welcomes voluntary contributions from any open-source enthusiast. If you want to make a contribution to this project, feel free to contact the development team.

## Contact

Regarding any questions, bugs, developments, or collaborations, please contact:

- Alessandro Traversa: traversa942@gmail.com
- Umair Bin Waheed: umair.waheed@kfupm.edu.sa

## Acknowledgments

We acknowledge the support provided by King Fahd University of Petroleum and Minerals (KFUPM) through the International Summer Program Research Grant no. ISP24201.
