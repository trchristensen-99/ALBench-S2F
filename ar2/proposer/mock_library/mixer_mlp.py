import torch
import torch.nn as nn

HPARAMS = {"dim": 64, "depth": 3, "patch": 4, "dropout": 0.1}
TUNABLE = {"dim": [32, 128], "depth": [2, 6], "dropout": [0.0, 0.3]}


class TokenMixer(nn.Module):
    def __init__(self, tokens: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(tokens)
        self.mlp = nn.Sequential(
            nn.Linear(tokens, 2 * tokens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * tokens, tokens),
        )

    def forward(self, x):  # x: (B, T, D); mixes along T
        h = self.norm(x.transpose(1, 2))
        return x + self.mlp(h).transpose(1, 2)


class ChannelMixer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim)
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class Net(nn.Module):
    def __init__(self, out_dim: int, dim: int, depth: int, patch: int, dropout: float):
        super().__init__()
        self.patch = patch
        tokens = 200 // patch
        self.embed = nn.Linear(4 * patch, dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(TokenMixer(tokens, dropout))
            self.layers.append(ChannelMixer(dim, dropout))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x):
        b = x.shape[0]
        t = 200 // self.patch
        h = self.embed(
            x.reshape(b, 4, t, self.patch).permute(0, 2, 1, 3).reshape(b, t, 4 * self.patch)
        )
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h).mean(dim=1))


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["dim"]), int(hp["depth"]), int(hp["patch"]), float(hp["dropout"]))
