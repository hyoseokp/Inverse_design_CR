from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        groups: int = 8,
        circular: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            padding_mode="circular" if circular else "zeros",
            bias=False,
        )
        self.gn = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int, *, groups: int = 8, circular: bool = True) -> None:
        super().__init__()
        self.c1 = ConvGNAct(ch, ch, groups=groups, circular=circular)
        self.c2 = nn.Conv2d(
            ch,
            ch,
            3,
            padding=1,
            padding_mode="circular" if circular else "zeros",
            bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(x)
        h = self.gn2(self.c2(h))
        return F.silu(x + h)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, groups: int = 8, circular: bool = True) -> None:
        super().__init__()
        self.c1 = ConvGNAct(in_ch, out_ch, s=2, groups=groups, circular=circular)  # stride-2 downsample
        self.r1 = ResBlock(out_ch, groups=groups, circular=circular)
        self.r2 = ResBlock(out_ch, groups=groups, circular=circular)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.r2(self.r1(x))
        return x


class Up(nn.Module):
    def __init__(
        self, in_ch: int, *, skip_ch: int, out_ch: int, groups: int = 8, circular: bool = True
    ) -> None:
        super().__init__()
        self.pre = ConvGNAct(in_ch, out_ch, groups=groups, circular=circular)
        self.fuse = ConvGNAct(out_ch + skip_ch, out_ch, groups=groups, circular=circular) if skip_ch > 0 else None
        self.r1 = ResBlock(out_ch, groups=groups, circular=circular)
        self.r2 = ResBlock(out_ch, groups=groups, circular=circular)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.pre(x)
        if self.fuse is not None and skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.fuse(x)
        x = self.r2(self.r1(x))
        return x


class ResUNet16to128(nn.Module):
    """16x16 -> 128x128 predictor (from user's notebook).

    Input:  (B,1,16,16) seed in [0,1]
    Output: (B,1,128,128) logits by default
    """

    def __init__(
        self,
        *,
        base: int = 64,
        groups: int = 8,
        circular: bool = True,
        enforce_diag_sym: bool = True,
        return_logits: bool = True,
    ) -> None:
        super().__init__()
        self.enforce_diag_sym = bool(enforce_diag_sym)
        self.return_logits = bool(return_logits)

        self.stem = nn.Sequential(
            ConvGNAct(1, base, groups=groups, circular=circular),
            ResBlock(base, groups=groups, circular=circular),
        )

        self.d1 = Down(base, base * 2, groups=groups, circular=circular)  # 8x8
        self.d2 = Down(base * 2, base * 4, groups=groups, circular=circular)  # 4x4
        self.d3 = Down(base * 4, base * 4, groups=groups, circular=circular)  # 2x2

        self.bot = nn.Sequential(
            ResBlock(base * 4, groups=groups, circular=circular),
            ResBlock(base * 4, groups=groups, circular=circular),
        )

        self.u3 = Up(base * 4, skip_ch=base * 4, out_ch=base * 4, groups=groups, circular=circular)  # 4
        self.u2 = Up(base * 4, skip_ch=base * 2, out_ch=base * 2, groups=groups, circular=circular)  # 8
        self.u1 = Up(base * 2, skip_ch=base, out_ch=base, groups=groups, circular=circular)  # 16

        self.u16_32 = Up(base, skip_ch=0, out_ch=base // 2, groups=groups, circular=circular)  # 32
        self.u32_64 = Up(base // 2, skip_ch=0, out_ch=base // 4, groups=groups, circular=circular)  # 64
        self.u64_128 = Up(base // 4, skip_ch=0, out_ch=base // 4, groups=groups, circular=circular)  # 128

        self.head = nn.Conv2d(
            base // 4,
            1,
            kernel_size=3,
            padding=1,
            padding_mode="circular" if circular else "zeros",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)  # 16
        s1 = self.d1(s0)  # 8
        s2 = self.d2(s1)  # 4
        s3 = self.d3(s2)  # 2

        b = self.bot(s3)

        x = self.u3(b, s2)  # 4
        x = self.u2(x, s1)  # 8
        x = self.u1(x, s0)  # 16

        x = self.u16_32(x)  # 32
        x = self.u32_64(x)  # 64
        x = self.u64_128(x)  # 128

        logits = self.head(x)

        if self.enforce_diag_sym:
            z = logits[:, 0]
            z = 0.5 * (z + z.transpose(-1, -2))
            logits = z[:, None]

        if self.return_logits:
            return logits
        return torch.sigmoid(logits)

