import contextlib
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import plotext as plt


def _amp_backend(device: torch.device):
    if device.type in ("cuda", "mps"):
        return device.type
    return None


def _make_scaler(device: torch.device, use_amp: bool):
    if not use_amp:
        return None
    b = _amp_backend(device)
    return torch.amp.GradScaler(b) if b else None


@contextlib.contextmanager
def _autocast(device: torch.device, use_amp: bool):
    b = _amp_backend(device)
    if use_amp and b is not None:
        with torch.autocast(device_type=b, dtype=torch.float16):
            yield
    else:
        yield


def _save(images, path, nrow=8):
    save_image(images * 0.5 + 0.5, path, nrow=nrow)


def _spatial_to_core(x, *, data_size: int, core_size: int):
    """Downsample batch (N,C,H,W) to core_size when data resolution > model core."""
    if data_size == core_size:
        return x
    return F.interpolate(
        x, size=(core_size, core_size), mode="bilinear", align_corners=False
    )


def _spatial_to_display(x, *, data_size: int, core_size: int):
    """Upsample model output to data/GIF resolution."""
    if data_size == core_size:
        return x
    return F.interpolate(
        x, size=(data_size, data_size), mode="bilinear", align_corners=False
    )


# ── In-place terminal chart ──────────────────────────────────────────

_prev_chart_h = 0
# Cursor-up erase must cover plotext output *plus* tqdm (often 1–2 terminal rows if it wraps).
_ERASE_EXTRA_LINES = 6


def _terminal_width():
    try:
        return max(50, min(shutil.get_terminal_size((88, 20)).columns - 2, 100))
    except OSError:
        return 88


def _epoch_pbar(it, desc):
    """Epoch bar with bounded width so bar+postfix usually stay on one row (helps in-place chart)."""
    return tqdm(
        it,
        desc=desc,
        unit="ep",
        ncols=_terminal_width(),
        dynamic_ncols=False,
        mininterval=0.3,
    )


def _plot_inplace(*series, title="", yscale="linear"):
    """Redraw a terminal chart in-place, overwriting the previous one."""
    global _prev_chart_h
    plt.clear_figure()
    plt.theme("dark")
    plt.plot_size(_terminal_width(), 15)
    plt.yscale(yscale)
    for ys, label in series:
        if yscale == "log":
            ys = [max(float(v), 1e-8) for v in ys]
        plt.plot(ys, label=label)
    plt.title(title)
    plt.xlabel("epoch")
    chart = plt.build()

    new_h = len(chart.splitlines())
    if new_h == 0:
        new_h = 1
    # Underestimate if the terminal soft-wraps wide ANSI lines; extra lines in erase helps.
    up = _prev_chart_h + _ERASE_EXTRA_LINES if _prev_chart_h else 0
    erase = f"\033[{up}A\033[J" if up else ""
    tqdm.write(erase + chart)
    _prev_chart_h = new_h


def _reset_chart():
    global _prev_chart_h
    _prev_chart_h = 0


# ── Run tracking ─────────────────────────────────────────────────────

_FRAME_SCALE = 4
_FRAME_GAP = 4
_N_SEEDS = 5


class RunTracker:
    """Keeps per-run artefacts: metric log, per-epoch GIF frames, sample grids."""

    def __init__(self, run_dir, image_size=32):
        self.run_dir = run_dir
        self.frame_dir = os.path.join(run_dir, "frames")
        self.sample_dir = os.path.join(run_dir, "samples")
        os.makedirs(self.frame_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        self.image_size = image_size
        self._log = open(os.path.join(run_dir, "log.txt"), "w")
        self._frames: list[Image.Image] = []

    def log(self, line: str):
        self._log.write(line + "\n")
        self._log.flush()

    def save_frame(self, epoch: int, images: torch.Tensor):
        """Render *images* (N,C,H,W in [-1,1]) side-by-side, upscaled, as one frame."""
        n, c = images.size(0), images.size(1)
        raw = ((images.clamp(-1, 1) * 0.5 + 0.5) * 255).byte().cpu()
        cell = self.image_size * _FRAME_SCALE
        w = n * cell + (n - 1) * _FRAME_GAP
        mode = "RGB" if c == 3 else "L"
        canvas = Image.new(mode, (w, cell), (255, 255, 255) if c == 3 else 255)
        for i in range(n):
            if c == 3:
                pil = Image.fromarray(raw[i].permute(1, 2, 0).numpy(), mode="RGB")
            else:
                pil = Image.fromarray(raw[i, 0].numpy(), mode="L")
            pil = pil.resize((cell, cell), Image.NEAREST)
            canvas.paste(pil, (i * (cell + _FRAME_GAP), 0))
        canvas.save(os.path.join(self.frame_dir, f"{epoch:04d}.png"))
        self._frames.append(canvas)

    def finish(self):
        self._build_gif()
        self._log.close()

    def _build_gif(self):
        if not self._frames:
            return
        path = os.path.join(self.run_dir, "progress.gif")
        # ~75ms per epoch frame (2× faster than 150ms); final frame holds 1s
        durations = [75] * (len(self._frames) - 1) + [1000]
        self._frames[0].save(
            path,
            save_all=True,
            append_images=self._frames[1:],
            duration=durations,
            loop=0,
        )
        tqdm.write(f"GIF saved → {path}  ({len(self._frames)} frames)")


# ── GAN ──────────────────────────────────────────────────────────────

def train_gan(gen, disc, loader, device, *, epochs, lr, z_dim, tracker,
              data_size=32, core_size=32, use_amp=False):
    _reset_chart()
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    crit = nn.BCEWithLogitsLoss()
    scaler = _make_scaler(device, use_amp)
    fixed_z = torch.randn(64, z_dim, device=device)
    seeds = torch.randn(_N_SEEDS, z_dim, device=device)

    hist_d, hist_g = [], []
    bar = _epoch_pbar(range(1, epochs + 1), "[GAN]")

    try:
        for ep in bar:
            gl = dl = 0.0
            for real, _ in loader:
                bs = real.size(0)
                real = real.to(device)
                real_c = _spatial_to_core(real, data_size=data_size, core_size=core_size)

                opt_d.zero_grad(set_to_none=True)
                with _autocast(device, use_amp):
                    z = torch.randn(bs, z_dim, device=device)
                    fake = gen(z).detach()
                    d_loss = crit(disc(real_c), torch.ones(bs, device=device)) + \
                             crit(disc(fake), torch.zeros(bs, device=device))
                if scaler:
                    scaler.scale(d_loss).backward()
                    scaler.step(opt_d)
                    scaler.update()
                else:
                    d_loss.backward()
                    opt_d.step()

                opt_g.zero_grad(set_to_none=True)
                with _autocast(device, use_amp):
                    z = torch.randn(bs, z_dim, device=device)
                    g_loss = crit(disc(gen(z)), torch.ones(bs, device=device))
                if scaler:
                    scaler.scale(g_loss).backward()
                    scaler.step(opt_g)
                    scaler.update()
                else:
                    g_loss.backward()
                    opt_g.step()

                dl += d_loss.item(); gl += g_loss.item()

            n = len(loader)
            hist_d.append(dl / n); hist_g.append(gl / n)
            bar.set_postfix(D=f"{hist_d[-1]:.4f}", G=f"{hist_g[-1]:.4f}")
            tracker.log(f"epoch={ep}  D_loss={hist_d[-1]:.4f}  G_loss={hist_g[-1]:.4f}")

            with torch.no_grad():
                with _autocast(device, use_amp):
                    g_out = _spatial_to_display(
                        gen(seeds), data_size=data_size, core_size=core_size)
                tracker.save_frame(ep, g_out)
                if ep % 10 == 0 or ep == 1:
                    with _autocast(device, use_amp):
                        grid = _spatial_to_display(
                            gen(fixed_z), data_size=data_size, core_size=core_size)
                    _save(grid, f"{tracker.sample_dir}/epoch_{ep:04d}.png")

            _plot_inplace((hist_d, "D_loss"), (hist_g, "G_loss"), title="GAN training")
    finally:
        tracker.finish()


# ── VAE ──────────────────────────────────────────────────────────────

def train_vae(enc, dec, loader, device, *, epochs, lr, latent_dim, tracker,
              data_size=32, core_size=32, use_amp=False):
    _reset_chart()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    scaler = _make_scaler(device, use_amp)
    fixed_z = torch.randn(64, latent_dim, device=device)
    seeds = torch.randn(_N_SEEDS, latent_dim, device=device)

    hist_r, hist_kl = [], []
    bar = _epoch_pbar(range(1, epochs + 1), "[VAE]")

    try:
        for ep in bar:
            tl = tr = tk = 0.0
            for real, _ in loader:
                real = real.to(device)
                real_c = _spatial_to_core(real, data_size=data_size, core_size=core_size)
                opt.zero_grad(set_to_none=True)
                with _autocast(device, use_amp):
                    mu, logvar = enc(real_c)
                    z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
                    recon = dec(z)
                    recon_loss = F.mse_loss(recon, real_c, reduction="sum") / real.size(0)
                    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / real.size(0)
                    loss = recon_loss + kl
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                tl += loss.item(); tr += recon_loss.item(); tk += kl.item()

            n = len(loader)
            hist_r.append(tr / n); hist_kl.append(tk / n)
            bar.set_postfix(R=f"{hist_r[-1]:.1f}", KL=f"{hist_kl[-1]:.1f}")
            tracker.log(f"epoch={ep}  loss={tl/n:.1f}  recon={hist_r[-1]:.1f}  KL={hist_kl[-1]:.1f}")

            with torch.no_grad():
                with _autocast(device, use_amp):
                    d_out = _spatial_to_display(
                        dec(seeds), data_size=data_size, core_size=core_size)
                tracker.save_frame(ep, d_out)
                if ep % 10 == 0 or ep == 1:
                    with _autocast(device, use_amp):
                        grid = _spatial_to_display(
                            dec(fixed_z), data_size=data_size, core_size=core_size)
                    _save(grid, f"{tracker.sample_dir}/epoch_{ep:04d}.png")

            _plot_inplace(
                (hist_r, "Recon"),
                (hist_kl, "KL"),
                title="VAE training (log y)",
                yscale="log",
            )
    finally:
        tracker.finish()


# ── Diffusion ────────────────────────────────────────────────────────

class NoiseSchedule:
    def __init__(self, T=1000, device="cpu"):
        betas = torch.linspace(1e-4, 0.02, T, device=device)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        self.T = T
        self.betas = betas
        self.alphas = alphas
        self.abar = abar
        self.sqrt_abar = abar.sqrt()
        self.sqrt_1m_abar = (1 - abar).sqrt()

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_abar[t][:, None, None, None]
        b = self.sqrt_1m_abar[t][:, None, None, None]
        return a * x0 + b * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x, t_idx):
        t = torch.full((x.size(0),), t_idx, device=x.device, dtype=torch.long)
        eps = model(x, t)
        alpha = self.alphas[t_idx]
        abar = self.abar[t_idx]
        mean = (1 / alpha.sqrt()) * (x - self.betas[t_idx] / (1 - abar).sqrt() * eps)
        if t_idx > 0:
            return mean + self.betas[t_idx].sqrt() * torch.randn_like(x)
        return mean

    @torch.no_grad()
    def sample_loop(self, model, shape, device):
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)


def train_diffusion(model, loader, device, *, epochs, lr, T, img_ch=1,
                     tracker, data_size=32, core_size=32, use_amp=False):
    _reset_chart()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = _make_scaler(device, use_amp)
    sched = NoiseSchedule(T=T, device=device)
    seed_noise = torch.randn(_N_SEEDS, img_ch, core_size, core_size,
                             device=device)

    hist = []
    bar = _epoch_pbar(range(1, epochs + 1), "[Diff]")

    try:
        for ep in bar:
            tl = 0.0
            for real, _ in loader:
                real = real.to(device)
                real_c = _spatial_to_core(real, data_size=data_size, core_size=core_size)
                t = torch.randint(0, T, (real_c.size(0),), device=device)
                noisy, noise = sched.q_sample(real_c, t)
                opt.zero_grad(set_to_none=True)
                with _autocast(device, use_amp):
                    loss = F.mse_loss(model(noisy, t), noise)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                tl += loss.item()

            n = len(loader)
            hist.append(tl / n)
            bar.set_postfix(loss=f"{hist[-1]:.4f}")
            tracker.log(f"epoch={ep}  loss={hist[-1]:.4f}")

            with torch.no_grad():
                with _autocast(device, use_amp):
                    x = seed_noise.clone()
                    for i in reversed(range(T)):
                        x = sched.p_sample(model, x, i)
                frame = _spatial_to_display(
                    x.clamp(-1, 1), data_size=data_size, core_size=core_size)
                tracker.save_frame(ep, frame)

                if ep % 10 == 0 or ep == 1:
                    with _autocast(device, use_amp):
                        imgs = sched.sample_loop(
                            model, (64, img_ch, core_size, core_size), device)
                    _save(
                        _spatial_to_display(
                            imgs, data_size=data_size, core_size=core_size),
                        f"{tracker.sample_dir}/epoch_{ep:04d}.png",
                    )

            _plot_inplace((hist, "noise_pred_loss"), title="Diffusion training")
    finally:
        tracker.finish()
