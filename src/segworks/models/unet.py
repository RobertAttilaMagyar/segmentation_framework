from collections.abc import Iterable

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, batch_norm: bool = True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 3, padding=1),
            nn.BatchNorm2d(out_dims) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims, 3, padding=1),
            nn.BatchNorm2d(out_dims) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.down = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        return (skip_connection_x := self.layers(x)), self.down(skip_connection_x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_dims: int,
        skip_dims: int,
        out_dims: int,
        batch_norm: bool = True,
        bilinear: bool = False,
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_dims, in_dims, 2, stride=2)

        self.layers = nn.Sequential(
            nn.Conv2d(in_dims + skip_dims, out_dims, 3, padding=1),
            nn.BatchNorm2d(out_dims) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims, 3, padding=1),
            nn.BatchNorm2d(out_dims) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        return self.layers(x)


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 2 * out_channels, 3, padding=1),
            nn.BatchNorm2d(2 * out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        levels: int = 5,
        inter_channel_counts: Iterable[int] | None = None,
    ):
        super().__init__()

        if inter_channel_counts is None:
            inter_channel_counts = [2 ** (5 + i) for i in range(levels)]

        if len(inter_channel_counts) != levels:
            raise ValueError("inter_channel_counts must match levels")

        self.down_modules = nn.ModuleList()
        self.down_modules.append(EncoderBlock(in_channels, inter_channel_counts[0]))
        for c1, c2 in zip(inter_channel_counts[:-1], inter_channel_counts[1:], strict=False):
            self.down_modules.append(EncoderBlock(c1, c2))

        self.bottleneck = BottleNeck(
            inter_channel_counts[-1],
            inter_channel_counts[-1],
        )

        self.up_modules = nn.ModuleList()
        rev_channels = list(reversed(inter_channel_counts))
        rev_channels.append(inter_channel_counts[0])
        for in_c, skip_c, out_c in zip(
            rev_channels,
            rev_channels,
            rev_channels[1:],
            strict=False,
        ):
            self.up_modules.append(
                DecoderBlock(
                    in_dims=in_c,
                    skip_dims=skip_c,
                    out_dims=out_c,
                )
            )

        self.out_conv = nn.Conv2d(inter_channel_counts[0], out_channels, 1)

    def forward(self, x: torch.Tensor):
        skips = []

        for down in self.down_modules:
            skip, x = down(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for up in self.up_modules:
            skip = skips.pop()
            x = up(x, skip)

        return self.out_conv(x)
