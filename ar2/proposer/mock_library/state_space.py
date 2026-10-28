import torch
import torch.nn as nn
import torch.nn.functional as F

HPARAMS = {"width": 48, "depth": 3, "state": 16}
TUNABLE = {"width": [24, 96], "depth": [2, 5], "state": [8, 32]}


class SSMLayer(nn.Module):
    """Diagonal linear state-space layer run as a long convolution: kernel[t] = sum_n C_n * exp(-a_n t) * B_n, applied in both directions."""

    def __init__(self, ch: int, state: int):
        super().__init__()
        self.log_a = nn.Parameter(torch.randn(ch, state) * 0.5 - 2.0)
        self.b = nn.Parameter(torch.randn(ch, state) * 0.3)
        self.c = nn.Parameter(torch.randn(ch, state) * 0.3)
        self.skip = nn.Parameter(torch.ones(ch))
        self.out = nn.Conv1d(2 * ch, ch, 1)
        self.norm = nn.GroupNorm(1, ch)

    def kernel(self, length: int):
        t = torch.arange(length, device=self.b.device, dtype=self.b.dtype)
        decay = torch.exp(-torch.exp(self.log_a).unsqueeze(-1) * t)  # (ch, state, L)
        return torch.einsum("cn,cnl->cl", self.c * self.b, decay)

    def causal(self, x, k):
        length = x.shape[-1]
        return F.conv1d(F.pad(x, (length - 1, 0)), k.flip(-1).unsqueeze(1), groups=x.shape[1])

    def forward(self, x):
        k = self.kernel(x.shape[-1]).to(x.dtype)
        fwd = self.causal(x, k)
        bwd = self.causal(x.flip(-1), k).flip(-1)
        y = self.out(torch.cat([fwd, bwd], dim=1)) + self.skip.view(1, -1, 1) * x
        return F.gelu(self.norm(y))


class Net(nn.Module):
    def __init__(self, out_dim: int, width: int, depth: int, state: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, width, 9, padding=4), nn.GroupNorm(1, width), nn.GELU()
        )
        self.layers = nn.ModuleList([SSMLayer(width, state) for _ in range(depth)])
        self.head = nn.Linear(2 * width, out_dim)

    def forward(self, x):
        h = self.stem(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))


def build_model(out_dim: int, **hparams) -> nn.Module:
    hp = {**HPARAMS, **hparams}
    return Net(out_dim, int(hp["width"]), int(hp["depth"]), int(hp["state"]))
