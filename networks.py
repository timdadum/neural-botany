import math
import torch
import torch.nn as nn


def _spatial_depth(image_size: int) -> int:
    """Number of ×2 stages for power-of-2 sizes (32→3, 64→4)."""
    if image_size < 32 or image_size & (image_size - 1):
        raise ValueError("internal: not a power-of-2 size")
    return int(math.log2(image_size)) - 2


def _unet_depth(image_size: int) -> int:
    """Down/up stages for UNet. 48×48 uses the same depth as 64 (bottleneck 3×3 vs 4×4)."""
    if image_size == 48:
        return 4
    return _spatial_depth(image_size)


def model_spatial_size(data_image_size: int) -> int:
    """Training resolution = model resolution (no hidden resize)."""
    if data_image_size < 32:
        raise ValueError("image_size must be >= 32")
    if data_image_size == 48:
        return 48
    if data_image_size >= 32 and (data_image_size & (data_image_size - 1)) == 0:
        return data_image_size
    raise ValueError("image_size must be 48 or a power of 2 (32, 64, 128, …)")


# ── shared conv building blocks ──────────────────────────────────────

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def _init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ── GAN ──────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """Pow2: 4×4 … → S×S. Size 48: 1×1 → 3×3 → … → 48 (four ×2 ups)."""

    def __init__(self, z_dim=100, base_ch=48, img_ch=1, image_size=32):
        super().__init__()
        if image_size == 48:
            h = base_ch * 8
            self.net = nn.Sequential(
                nn.ConvTranspose2d(z_dim, h, 3, 1, 0, bias=False),
                nn.BatchNorm2d(h),
                nn.ReLU(True),
                UpBlock(h, base_ch * 4),
                UpBlock(base_ch * 4, base_ch * 2),
                UpBlock(base_ch * 2, base_ch),
                nn.ConvTranspose2d(base_ch, img_ch, 4, 2, 1),
                nn.Tanh(),
            )
        else:
            n = _spatial_depth(image_size)
            first_ch = base_ch * (2 ** (n - 1))
            layers = [
                nn.ConvTranspose2d(z_dim, first_ch, 4, 1, 0, bias=False),
                nn.BatchNorm2d(first_ch),
                nn.ReLU(True),
            ]
            for k in range(n - 1):
                inc = base_ch * (2 ** (n - 1 - k))
                outc = base_ch * (2 ** (n - 2 - k))
                layers.append(UpBlock(inc, outc))
            layers += [nn.ConvTranspose2d(base_ch, img_ch, 4, 2, 1), nn.Tanh()]
            self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, img_ch=1, base_ch=48, image_size=32):
        super().__init__()
        if image_size == 48:
            layers = [
                DownBlock(img_ch, base_ch, use_bn=False),
                DownBlock(base_ch, base_ch * 2),
                DownBlock(base_ch * 2, base_ch * 4),
                DownBlock(base_ch * 4, base_ch * 8),
                nn.Conv2d(base_ch * 8, 1, 3, 1, 0),
            ]
            self.net = nn.Sequential(*layers)
        else:
            n = _spatial_depth(image_size)
            layers = []
            c_in = img_ch
            for i in range(n):
                c_out = base_ch * (2 ** i)
                layers.append(DownBlock(c_in, c_out, use_bn=(i > 0)))
                c_in = c_out
            layers.append(nn.Conv2d(c_in, 1, 4, 1, 0))
            self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x):
        return self.net(x).view(-1)


# ── VAE ──────────────────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    def __init__(self, img_ch=1, base_ch=48, latent_dim=128, image_size=32):
        super().__init__()
        if image_size == 48:
            self.features = nn.Sequential(
                DownBlock(img_ch, base_ch, use_bn=False),
                DownBlock(base_ch, base_ch * 2),
                DownBlock(base_ch * 2, base_ch * 4),
                DownBlock(base_ch * 4, base_ch * 8),
            )
            flat = base_ch * 8 * 3 * 3
        else:
            n = _spatial_depth(image_size)
            blocks = []
            c_in = img_ch
            for i in range(n):
                c_out = base_ch * (2 ** i)
                blocks.append(DownBlock(c_in, c_out, use_bn=(i > 0)))
                c_in = c_out
            self.features = nn.Sequential(*blocks)
            flat = c_in * 4 * 4
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128, base_ch=48, img_ch=1, image_size=32):
        super().__init__()
        self.base_ch = base_ch
        if image_size == 48:
            h = base_ch * 8
            self.hidden_ch = h
            self._bottleneck_side = 3
            self.project = nn.Sequential(
                nn.Linear(latent_dim, h * 3 * 3),
                nn.ReLU(True),
            )
            self.net = nn.Sequential(
                UpBlock(h, base_ch * 4),
                UpBlock(base_ch * 4, base_ch * 2),
                UpBlock(base_ch * 2, base_ch),
                nn.ConvTranspose2d(base_ch, img_ch, 4, 2, 1),
                nn.Tanh(),
            )
        else:
            n = _spatial_depth(image_size)
            hidden = base_ch * (2 ** (n - 1))
            self.hidden_ch = hidden
            self._bottleneck_side = 4
            self.project = nn.Sequential(
                nn.Linear(latent_dim, hidden * 4 * 4),
                nn.ReLU(True),
            )
            layers = []
            for k in range(n - 1):
                inc = base_ch * (2 ** (n - 1 - k))
                outc = base_ch * (2 ** (n - 2 - k))
                layers.append(UpBlock(inc, outc))
            layers += [nn.ConvTranspose2d(base_ch, img_ch, 4, 2, 1), nn.Tanh()]
            self.net = nn.Sequential(*layers)

    def forward(self, z):
        s = self._bottleneck_side
        h = self.project(z).view(-1, self.hidden_ch, s, s)
        return self.net(h)


# ── Diffusion UNet ───────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = math.log(10_000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -f)
        emb = t.float().unsqueeze(1) * f.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=1)


class UNet(nn.Module):
    def __init__(self, img_ch=1, base_ch=48, time_dim=128, image_size=32):
        super().__init__()
        self.image_size = image_size
        n = _unet_depth(image_size)

        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )

        self.downs = nn.ModuleList()
        self.t_down = nn.ModuleList()
        c_in = img_ch
        for i in range(n):
            c_out = base_ch * (2 ** i)
            self.downs.append(self._block(c_in, c_out))
            self.t_down.append(nn.Linear(time_dim, c_out))
            c_in = c_out

        c_bot = base_ch * (2 ** (n - 1))
        self.bot = self._block(c_bot, c_bot)
        self.t_bot = nn.Linear(time_dim, c_bot)
        self.pool = nn.MaxPool2d(2)

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for k in range(n):
            uc = base_ch * (2 ** (n - 1 - k))
            self.ups.append(nn.ConvTranspose2d(uc, uc, 2, 2))
            dec_in = uc * 2
            dec_out = base_ch * (2 ** (n - 2 - k)) if k < n - 1 else base_ch
            self.decs.append(self._block(dec_in, dec_out))

        self.out = nn.Conv2d(base_ch, img_ch, 1)
        self.apply(_init_weights)

    @staticmethod
    def _block(in_ch, out_ch):
        gn = min(8, out_ch)
        if out_ch % gn != 0:
            gn = 1
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(gn, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(gn, out_ch),
            nn.GELU(),
        )

    def forward(self, x, t):
        te = self.time_mlp(t)
        ds = []
        for i in range(len(self.downs)):
            if i > 0:
                x = self.pool(x)
            x = self.downs[i](x) + self.t_down[i](te)[:, :, None, None]
            ds.append(x)
        x = self.pool(ds[-1])
        u = self.bot(x) + self.t_bot(te)[:, :, None, None]
        n = len(self.ups)
        for k in range(n):
            u = self.ups[k](u)
            skip = ds[-1 - k]
            u = self.decs[k](torch.cat([u, skip], dim=1))
        return self.out(u)


