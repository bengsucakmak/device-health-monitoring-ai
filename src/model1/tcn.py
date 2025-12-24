from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # same-length
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNDisaggregator(nn.Module):
    """
    Lightweight TCN-like model:
      input:  (B, 1, L)
      output: (B, 1, L)
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.ReLU(),
        )

        blocks = []
        # dilations: 1,2,4,8,...
        for i in range(num_blocks):
            blocks.append(ResidualBlock(hidden, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)

        self.out_proj = nn.Conv1d(hidden, 1, kernel_size=1)

        # Negatif güç çıkmasını istemiyoruz -> ReLU
        self.out_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.tcn(h)
        y = self.out_proj(h)
        return self.out_act(y)
