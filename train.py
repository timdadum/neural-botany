#!/usr/bin/env python3
"""Train a generative model (gan / vae / diffusion) on Oxford 102 Flowers.

Usage:
    python train.py --model gan       [--epochs 100] [--lr 2e-4] [--color]
    python train.py --model vae       [--epochs 100] [--lr 1e-3] [--color]
    python train.py --model diffusion [--epochs 100] [--lr 1e-4] [--color]

Each run is saved to  runs/<model>_<timestamp>/  containing:
    checkpoint.pt   trained weights
    progress.gif    5 fixed seeds evolving across epochs
    log.txt         per-epoch metrics
    samples/        64-image grids every 10 epochs
    frames/         individual GIF frames as PNGs
"""

import argparse
import os
from datetime import datetime
import torch
from data import get_dataloader
from networks import (
    Generator,
    Discriminator,
    VAEEncoder,
    VAEDecoder,
    UNet,
    model_spatial_size,
)
from trainers import train_gan, train_vae, train_diffusion, RunTracker


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("⚠  MPS not available, falling back to CPU")
    return torch.device("cpu")


def count(m):
    return f"{sum(p.numel() for p in m.parameters()):,}"


def save_checkpoint(path, model_type, state_dict, **config):
    torch.save({"model_type": model_type, "state_dict": state_dict, **config}, path)
    print(f"Checkpoint saved → {path}")


DEFAULTS = {
    "gan":       {"lr": 2e-4},
    "vae":       {"lr": 1e-3},
    "diffusion": {"lr": 1e-4},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gan", "vae", "diffusion"], required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--z-dim", type=int, default=100)
    p.add_argument("--image-size", type=int, default=48,
                   help="data & GIF resolution (48 or a power of 2 ≥ 32)")
    p.add_argument("--T", type=int, default=1000, help="diffusion timesteps")
    p.add_argument("--color", action="store_true", help="train on RGB instead of grayscale")
    p.add_argument("--base-ch", type=int, default=48,
                   help="conv width multiplier (smaller=faster; try 32–48)")
    p.add_argument("--no-amp", action="store_true",
                   help="disable float16 mixed precision (CUDA/MPS only)")
    args = p.parse_args()

    img_ch = 3 if args.color else 1
    lr = args.lr or DEFAULTS[args.model]["lr"]

    try:
        core = model_spatial_size(args.image_size)
    except ValueError as e:
        p.error(str(e))

    run_dir = f"runs/{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    dev = device()
    use_amp = not args.no_amp and dev.type in ("cuda", "mps")
    if use_amp:
        print("Mixed precision: float16 autocast + GradScaler")
    elif not args.no_amp and dev.type == "cpu":
        print("Mixed precision skipped (CPU)")
    print(f"Device: {dev}  |  base_ch={args.base_ch}")
    loader = get_dataloader(batch_size=args.batch_size, image_size=args.image_size,
                             color=args.color)

    tracker = RunTracker(run_dir, image_size=args.image_size)

    bc = args.base_ch

    if args.model == "gan":
        gen = Generator(z_dim=args.z_dim, base_ch=bc, img_ch=img_ch,
                        image_size=core).to(dev)
        disc = Discriminator(img_ch=img_ch, base_ch=bc, image_size=core).to(dev)
        print(f"Generator  {count(gen)} params  |  Discriminator  {count(disc)} params")
        train_gan(gen, disc, loader, dev, epochs=args.epochs, lr=lr, z_dim=args.z_dim,
                  tracker=tracker, data_size=args.image_size, core_size=core,
                  use_amp=use_amp)
        save_checkpoint(f"{run_dir}/checkpoint.pt", "gan",
                        gen.cpu().state_dict(), z_dim=args.z_dim, img_ch=img_ch,
                        image_size=args.image_size, model_core_size=core,
                        base_ch=bc)

    elif args.model == "vae":
        enc = VAEEncoder(latent_dim=args.z_dim, base_ch=bc, img_ch=img_ch,
                         image_size=core).to(dev)
        dec = VAEDecoder(latent_dim=args.z_dim, base_ch=bc, img_ch=img_ch,
                         image_size=core).to(dev)
        print(f"Encoder  {count(enc)} params  |  Decoder  {count(dec)} params")
        train_vae(enc, dec, loader, dev, epochs=args.epochs, lr=lr,
                  latent_dim=args.z_dim, tracker=tracker,
                  data_size=args.image_size, core_size=core, use_amp=use_amp)
        save_checkpoint(f"{run_dir}/checkpoint.pt", "vae",
                        dec.cpu().state_dict(), latent_dim=args.z_dim, img_ch=img_ch,
                        image_size=args.image_size, model_core_size=core, base_ch=bc)

    elif args.model == "diffusion":
        unet = UNet(img_ch=img_ch, base_ch=bc, image_size=core).to(dev)
        print(f"UNet  {count(unet)} params")
        train_diffusion(unet, loader, dev, epochs=args.epochs, lr=lr, T=args.T,
                        img_ch=img_ch, tracker=tracker,
                        data_size=args.image_size, core_size=core, use_amp=use_amp)
        save_checkpoint(f"{run_dir}/checkpoint.pt", "diffusion",
                        unet.cpu().state_dict(), T=args.T, img_ch=img_ch,
                        image_size=args.image_size, model_core_size=core, base_ch=bc)


if __name__ == "__main__":
    main()
