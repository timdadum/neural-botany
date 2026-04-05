![Example GIF](example.gif)

# neural-botany

Train small generative models (GAN, VAE, Diffusion) from scratch on the [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset, targeting small square images (default **48×48**; configurable via `--image-size`, grayscale or colour) on **Apple Silicon (MPS)**.

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

The dataset (~330 MB) is downloaded automatically on the first run.

## Training

Pick a model and run:

```bash
python train.py --model gan              # DCGAN, 100 epochs, grayscale
python train.py --model vae              # VAE,   100 epochs, grayscale
python train.py --model diffusion        # DDPM,  100 epochs, grayscale
python train.py --model gan --color      # same, but RGB
```

All three use MPS by default and fall back to CPU when unavailable.

### Options


| Flag           | Default            | Notes                                                                                                                                                                                |
| -------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--epochs`     | 100                |                                                                                                                                                                                      |
| `--batch-size` | 128                |                                                                                                                                                                                      |
| `--lr`         | 2e-4 / 1e-3 / 1e-4 | per-model defaults                                                                                                                                                                   |
| `--z-dim`      | 100                | latent dimension (GAN & VAE)                                                                                                                                                         |
| `--image-size` | 48                 | side length (≥ 32). Allowed: **48** or a power of two (32, 64, …). Each size has a matching native conv architecture (48 uses a 3×3 bottleneck; powers of two use a 4×4-style grid). |
| `--T`          | 1000               | diffusion timesteps                                                                                                                                                                  |
| `--color`      | off                | train on 3-channel RGB instead of grayscale                                                                                                                                          |
| `--base-ch`    | 48                 | conv width multiplier (try **32** for faster runs; **64** for heavier models)                                                                                                        |
| `--no-amp`     | off                | disable float16 mixed precision on CUDA/MPS                                                                                                                                          |


**Speed:** On GPU, training uses **automatic mixed precision** by default.

### What gets saved

Every run creates a timestamped directory under `runs/`:

```
runs/gan_20260324_160000/
├── checkpoint.pt       trained model weights
├── progress.gif        5 fixed seeds evolving across all epochs
├── log.txt             per-epoch metrics
├── samples/            64-image grids (every 10 epochs)
└── frames/             individual GIF frames as PNGs
```

The **progress GIF** tracks 5 fixed latent starting points across every epoch so you can watch the model learn.

## Generating new images

After training, generate new flower images from random latent vectors:

```bash
python sample.py --checkpoint runs/<run>/checkpoint.pt --n 16
```


| Flag     | Default                 | Notes                        |
| -------- | ----------------------- | ---------------------------- |
| `--n`    | 16                      | number of images to generate |
| `--out`  | `generated_<model>.png` | output path                  |
| `--nrow` | min(n, 8)               | images per row in the grid   |


### How sampling works per model

- **GAN** — draw `z ~ N(0, 1)` of size `z_dim`, pass through the Generator.
Instant (one forward pass).
- **VAE** — draw `z ~ N(0, 1)` of size `latent_dim`, pass through the Decoder.
Instant (one forward pass).
- **Diffusion** — start from pure noise `x_T ~ N(0, 1)`, iteratively denoise  
for `T` steps using the trained UNet.  Slower (1000 forward passes by  
default).

## Data pipeline

1. **Center-crop** to `min(width, height)`, anchored at the image center
2. **Resize** to `--image-size` (default 48)
3. **Grayscale** (single channel) — skipped when `--color` is used
4. **Normalize** to [-1, 1]

All three dataset splits (train / val / test → 8 189 images total) are merged
into a single training set.

## Architecture overview

The three models share convolutional building blocks (`DownBlock`, `UpBlock`)
defined in `networks.py`:

- **GAN** — `Generator` (transposed-conv decoder) + `Discriminator` (conv
classifier).  Trained adversarially with BCE loss.
- **VAE** — `VAEEncoder` (conv → μ, log σ²) + `VAEDecoder` (same architecture
as the Generator).  Trained with MSE reconstruction + KL divergence.
- **Diffusion** — small `UNet` with skip connections, sinusoidal time
embeddings, and GroupNorm.  Trained to predict noise (DDPM).

## File overview

```
train.py       CLI entry point — trains, saves run artefacts
sample.py      Load a checkpoint and generate N new images
data.py        Oxford 102 Flowers dataloader with center-crop (+ optional grayscale)
networks.py    All network architectures (shared blocks + model-specific)
trainers.py    Training loops, run tracker, diffusion noise schedule
```

