from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Input:  (B, L, 1)
    Output: (B, L, 1)
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        enc_out, (h_n, _) = self.encoder(x)           # enc_out: (B,L,H)
        h_last = enc_out[:, -1, :]                    # (B,H)
        z = self.to_latent(h_last)                    # (B,Z)

        # Decode: latent -> repeat across time
        h0 = self.from_latent(z)                      # (B,H)
        h0 = h0.unsqueeze(1).repeat(1, x.size(1), 1)  # (B,L,H)

        dec_out, _ = self.decoder(h0)                 # (B,L,H)
        y = self.out(dec_out)                         # (B,L,1)
        return y
