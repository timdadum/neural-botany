#!/usr/bin/env python3
"""Generate new flower images from a trained checkpoint.

Usage:
    python sample.py --checkpoint runs/<run>/checkpoint.pt --n 16
"""

import argparse
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from networks import Generator, VAEDecoder, UNet
from trainers import NoiseSchedule


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _core_and_display(ckpt):
    """Architecture size vs saved output resolution (48+32 checkpoints)."""
    disp = ckpt.get("image_size", 32)
    core = ckpt.get("model_core_size", disp)
    return core, disp


def load_checkpoint(path, dev):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model_type = ckpt["model_type"]

    img_ch = ckpt.get("img_ch", 1)
    core, _ = _core_and_display(ckpt)
    bc = ckpt.get("base_ch", 64)

    if model_type == "gan":
        model = Generator(z_dim=ckpt["z_dim"], base_ch=bc, img_ch=img_ch,
                          image_size=core)
        model.load_state_dict(ckpt["state_dict"])
        return model.to(dev).eval(), model_type, ckpt

    if model_type == "vae":
        model = VAEDecoder(latent_dim=ckpt["latent_dim"], base_ch=bc,
                           img_ch=img_ch, image_size=core)
        model.load_state_dict(ckpt["state_dict"])
        return model.to(dev).eval(), model_type, ckpt

    if model_type == "diffusion":
        model = UNet(img_ch=img_ch, base_ch=bc, image_size=core)
        model.load_state_dict(ckpt["state_dict"])
        return model.to(dev).eval(), model_type, ckpt

    raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def generate(model, model_type, ckpt, n, dev):
    core, display = _core_and_display(ckpt)

    if model_type == "gan":
        z = torch.randn(n, ckpt["z_dim"], device=dev)
        out = model(z)

    elif model_type == "vae":
        z = torch.randn(n, ckpt["latent_dim"], device=dev)
        out = model(z)

    elif model_type == "diffusion":
        T = ckpt["T"]
        img_ch = ckpt.get("img_ch", 1)
        sched = NoiseSchedule(T=T, device=dev)
        x = torch.randn(n, img_ch, core, core, device=dev)
        for t in reversed(range(T)):
            x = sched.p_sample(model, x, t)
        out = x.clamp(-1, 1)

    if display != core:
        out = F.interpolate(
            out, size=(display, display), mode="bilinear", align_corners=False
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    p.add_argument("--n", type=int, default=16, help="number of images to generate")
    p.add_argument("--out", default=None, help="output .png path (default: generated_<type>.png)")
    p.add_argument("--nrow", type=int, default=None, help="images per row in grid")
    args = p.parse_args()

    dev = device()
    print(f"Device: {dev}")

    model, model_type, ckpt = load_checkpoint(args.checkpoint, dev)
    print(f"Loaded {model_type} checkpoint from {args.checkpoint}")

    images = generate(model, model_type, ckpt, args.n, dev)

    nrow = args.nrow or min(args.n, 8)
    out = args.out or f"generated_{model_type}.png"
    save_image(images * 0.5 + 0.5, out, nrow=nrow)
    print(f"Saved {args.n} images → {out}")


if __name__ == "__main__":
    main()
