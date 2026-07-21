"""Neural network architectures for projection learning and diffusion."""

import torch
import torch.nn as nn
import numpy as np


class ProjectionNet(nn.Module):
    """MLP projection model for low-to-medium dimensional data."""

    def __init__(self, ambient_dim, hidden=100, mid=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ambient_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, mid),
            nn.SiLU(),
            nn.Linear(mid, hidden),
            nn.SiLU(),
            nn.Linear(hidden, ambient_dim),
        )

    def forward(self, x):
        return self.net(x)


class ProjectionUNet(nn.Module):
    """Time-independent U-Net projection model with configurable depth."""

    def __init__(self, image_size=28, channels=(32, 64, 128, 256), in_channels=1):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.channels = list(channels)
        n = len(self.channels)

        def num_groups(ch, target=32):
            g = min(target, ch)
            while ch % g:
                g -= 1
            return g

        self.enc_convs = nn.ModuleList()
        self.enc_norms = nn.ModuleList()
        enc_sizes = []
        cur = image_size
        for i in range(n):
            stride = 1 if i == 0 else 2
            cur = (cur - 3) // stride + 1
            enc_sizes.append(cur)
            in_ch = self.in_channels if i == 0 else self.channels[i - 1]
            self.enc_convs.append(
                nn.Conv2d(in_ch, self.channels[i], 3, stride=stride, bias=False))
            self.enc_norms.append(
                nn.GroupNorm(num_groups(self.channels[i], 4 if i == 0 else 32),
                             num_channels=self.channels[i]))

        self.dec_tconvs = nn.ModuleList()
        self.dec_norms = nn.ModuleList()
        for k in range(n - 1):
            in_ch_dec = self.channels[n - 1] if k == 0 \
                else 2 * self.channels[n - 1 - k]
            out_ch_dec = self.channels[n - 2 - k]
            src = enc_sizes[n - 1 - k]
            tgt = enc_sizes[n - 2 - k]
            op = tgt - ((src - 1) * 2 + 3)
            if not (0 <= op <= 1):
                raise ValueError(f"output_padding={op} at k={k}.")
            self.dec_tconvs.append(
                nn.ConvTranspose2d(in_ch_dec, out_ch_dec, 3, stride=2,
                                   bias=False, output_padding=op))
            self.dec_norms.append(
                nn.GroupNorm(num_groups(out_ch_dec), num_channels=out_ch_dec))
        self.dec_final = nn.ConvTranspose2d(
            2 * self.channels[0], self.in_channels, 3, stride=1)
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        N = x.shape[0] if x.dim() >= 2 else 1
        n = len(self.channels)
        h = x.reshape(N, self.in_channels, self.image_size, self.image_size)
        skips = []
        for i in range(n):
            h = self.act(self.enc_norms[i](self.enc_convs[i](h)))
            skips.append(h)
        h = skips[n - 1]
        for k in range(n - 1):
            inp = h if k == 0 else torch.cat([h, skips[n - 1 - k]], dim=1)
            h = self.act(self.dec_norms[k](self.dec_tconvs[k](inp)))
        h = self.dec_final(torch.cat([h, skips[0]], dim=1))
        return h.reshape(N, -1)


# ========================= Diffusion model components =========================


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2)
                              * scale, requires_grad=False)

    def forward(self, x):
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.dim() == 1:
            x = x[:, None]
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ScoreNetVector(nn.Module):
    """Time-dependent score model (MLP) for vector-valued data."""

    def __init__(self, ambient_dim, marginal_prob_std, embed_dim=256,
                 hidden_dims=(100, 200, 100)):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.linear1 = nn.Linear(ambient_dim, h1)
        self.dense1 = nn.Linear(embed_dim, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.dense2 = nn.Linear(embed_dim, h2)
        self.linear3 = nn.Linear(h2, h3)
        self.dense3 = nn.Linear(embed_dim, h3)
        self.linear4 = nn.Linear(h3, ambient_dim)
        self.act = nn.SiLU()
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        embed = self.act(self.embed(t))
        h = self.act(self.linear1(x) + self.dense1(embed))
        h = self.act(self.linear2(h) + self.dense2(embed))
        h = self.act(self.linear3(h) + self.dense3(embed))
        h = self.linear4(h)
        return h / self.marginal_prob_std(t)[:, None]


class _Dense(nn.Module):
    """A fully connected layer that reshapes outputs to (B, C, 1, 1)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """Time-dependent score-based U-Net for 28x28 single-channel images.

    Mirrors the architecture used in the MNIST diffusion section of
    ``ProjectionLearning.ipynb``.
    """

    def __init__(self, marginal_prob_std, channels=(32, 64, 128, 256),
                 embed_dim=256):
        super().__init__()
        c = list(channels)
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        # Encoder
        self.conv1 = nn.Conv2d(1, c[0], 3, stride=1, bias=False)
        self.dense1 = _Dense(embed_dim, c[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=c[0])
        self.conv2 = nn.Conv2d(c[0], c[1], 3, stride=2, bias=False)
        self.dense2 = _Dense(embed_dim, c[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=c[1])
        self.conv3 = nn.Conv2d(c[1], c[2], 3, stride=2, bias=False)
        self.dense3 = _Dense(embed_dim, c[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=c[2])
        self.conv4 = nn.Conv2d(c[2], c[3], 3, stride=2, bias=False)
        self.dense4 = _Dense(embed_dim, c[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=c[3])
        # Decoder
        self.tconv4 = nn.ConvTranspose2d(c[3], c[2], 3, stride=2, bias=False)
        self.dense5 = _Dense(embed_dim, c[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=c[2])
        self.tconv3 = nn.ConvTranspose2d(c[2] + c[2], c[1], 3, stride=2,
                                         bias=False, output_padding=1)
        self.dense6 = _Dense(embed_dim, c[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=c[1])
        self.tconv2 = nn.ConvTranspose2d(c[1] + c[1], c[0], 3, stride=2,
                                         bias=False, output_padding=1)
        self.dense7 = _Dense(embed_dim, c[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=c[0])
        self.tconv1 = nn.ConvTranspose2d(c[0] + c[0], 1, 3, stride=1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        if x.dim() == 2:
            x = x.view(x.shape[0], 1, 28, 28)
        embed = self.act(self.embed(t))
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1))
                                  + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1))
                                  + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))
        return h / self.marginal_prob_std(t)[:, None, None, None]
