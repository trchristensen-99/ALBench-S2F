"""DREAM-Attn written to the AutoResearch contract (a ``Net`` class without constructor arguments), so it trains in the same harness as MPRA-LegNet and every proposal.

DREAM-Attn is the Prix Fixe model built around the UnlockDNA team's core block (Rafi et al., Nature Biotechnology 2024). The blocks are the published ones from ``benchmarks/human/prixfixe`` of
de-Boer-Lab/random-promoter-dream-challenge-2022, composed as that repository's human notebook composes them: AutosomeFirstLayersBlock -> UnlockDNACoreBlock -> AutosomeFinalLayersBlock.

Read the paper's own verdict before reading ours: DREAM-Attn was the weakest of the three DREAM-optimised models, did not beat MPRAnn on their smallest dataset (56k sequences), and could not be trained at all on
their K562 ATAC-seq task because the attention block's memory grew too large with long inputs. Neither failure is fatal here -- our input is 240 bp, two orders of magnitude shorter than an ATAC window -- but it is
the reason to expect this member to earn its place by decorrelating from the convolutional members rather than by its own accuracy.

Deviations from the published composition, all forced by our harness:
  * ``in_channels=4`` in place of their 5 (bases plus a flag channel); ``AutosomeFirstLayersBlock`` takes it as a parameter.
  * ``seqsize=240``, the fixed input standard, in place of 230. This one is NOT cosmetic: ``UnlockDNACoreBlock`` holds a learned ``nn.Embedding(seqsize, channels)`` positional table, so the sequence length is
    baked into the parameter count and this model -- alone among the four references -- cannot be fed another length without being rebuilt. The harness always feeds 240, so that is the size built here.
  * A single output, which the final block already produces.

Two faithfulness notes, preserved as published rather than silently corrected:
  * ``ConformerSASwiGLULayer`` takes a ``num_heads`` argument and then constructs ``nn.MultiheadAttention(..., num_heads=4)`` regardless, so the notebook's ``num_heads=8`` has no effect. Kept, because this is
    the model that was benchmarked; a corrected variant would be a different architecture and belongs in the search, not in the reference bank.
  * The published core block reads ``self.device``, a property of the Prix Fixe base class that does not exist outside that framework. Replaced with the input tensor's device, which is what it resolved to.

The recipe is NOT fixed here: a published reference gets its own per-size tuning, because a LegNet-tuned recipe would evaluate this architecture at another model's optimum. Attention models are the likeliest of
the four to need a different peak learning rate or a warmup, so this is the reference whose own-recipe bracket matters most.
"""

SOURCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLULayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.swish = nn.SiLU()

    def forward(self, x):
        out, gate = torch.chunk(x, 2, dim=self.dim)
        return out * self.swish(gate)


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, embedding_dim, mult=4, rate=0.0, use_bias=True):
        super().__init__()
        swiglu_out = int(embedding_dim * mult / 2)
        self.layernorm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim * mult, bias=use_bias)
        self.swiglulayer = SwiGLULayer(dim=1)
        self.drop = nn.Dropout(rate)
        self.linear2 = nn.Linear(swiglu_out, embedding_dim, bias=use_bias)

    def forward(self, inputs):
        x = self.layernorm(inputs.transpose(1, 2))
        x = self.linear1(x)
        x = self.swiglulayer(x.transpose(1, 2))
        x = self.drop(x)
        x = self.linear2(x.transpose(1, 2))
        return self.drop(x.transpose(1, 2))


class ConformerSASwiGLULayer(nn.Module):
    """Conformer block: half-step feed-forward, depthwise conv, self-attention, half-step feed-forward."""

    def __init__(self, embedding_dim, ff_mult=4, kernel_size=15, rate=0.2, num_heads=4, use_bias=False):
        super().__init__()
        self.ff1 = FeedForwardSwiGLU(embedding_dim=embedding_dim, mult=ff_mult, rate=rate, use_bias=use_bias)
        self.layernorm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size, groups=embedding_dim, padding='same', bias=False),
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Dropout(rate),
        )
        self.layernorm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        # as published: num_heads is accepted and then ignored here
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        self.ff2 = FeedForwardSwiGLU(embedding_dim=embedding_dim, mult=ff_mult, rate=rate, use_bias=use_bias)

    def forward(self, x):
        x = x.float()
        x = x + 0.5 * self.ff1(x)
        x1 = x.transpose(1, 2)
        x1 = self.layernorm1(x1)
        x1 = x1.transpose(1, 2)
        x1 = x1 + self.conv(x1)
        x = x + x1
        x = x.transpose(1, 2)
        x = self.layernorm2(x)
        x = x + self.attn(x, x, x)[0]
        x = x.transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return x


class AutosomeFirstLayersBlock(nn.Module):
    """A single wide convolution with batch norm and SiLU."""

    def __init__(self, in_channels=4, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm1d(out_channels, momentum=0.1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class UnlockDNACoreBlock(nn.Module):
    """Learned positional embedding added to the channel axis, then n Conformer blocks. Fixed to one sequence length."""

    def __init__(self, in_channels=256, out_channels=256, seqsize=240, num_heads=8, kernel_size=15, rate=0.1, n_blocks=4):
        super().__init__()
        assert in_channels == out_channels, "the positional table is added to the input, so the widths must match"
        self.seqsize = seqsize
        self.n_blocks = n_blocks
        self.out_channels = out_channels
        self.blocks = nn.ModuleList([
            ConformerSASwiGLULayer(embedding_dim=in_channels, kernel_size=kernel_size, rate=rate, num_heads=num_heads)
            for _ in range(n_blocks)
        ])
        self.pos_embedding = nn.Embedding(seqsize, out_channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        pos = torch.arange(start=0, end=self.seqsize, step=1, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos.long())
        x = x.transpose(1, 2)
        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        return x


class AutosomeFinalLayersBlock(nn.Module):
    """Pointwise conv to 256 channels, global average pool, one linear."""

    def __init__(self, in_channels=256):
        super().__init__()
        self.mapper = nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1, padding='same')
        self.linear = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x):
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2)
        return self.linear(x)


class Net(nn.Module):
    """DREAM-Attn in the human-benchmark configuration (first conv 256 ks 7, 4 Conformer blocks ks 15 rate .1), built for the 240-bp input standard."""

    def __init__(self):
        super().__init__()
        self.first = AutosomeFirstLayersBlock(in_channels=4, out_channels=256)
        self.core = UnlockDNACoreBlock(in_channels=self.first.out_channels, out_channels=self.first.out_channels,
                                       seqsize=240, n_blocks=4, kernel_size=15, rate=0.1, num_heads=8)
        self.final = AutosomeFinalLayersBlock(in_channels=self.core.out_channels)

    def forward(self, x):
        x = self.first(x)
        x = self.core(x)
        x = self.final(x)
        return x.squeeze(-1)
'''
