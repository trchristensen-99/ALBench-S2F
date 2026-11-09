"""DREAM-RNN written to the AutoResearch contract (a ``Net`` class without constructor arguments), so it trains in the same harness as MPRA-LegNet and every proposal.

DREAM-RNN is the Prix Fixe model built around the BHI team's core block (Rafi et al., Nature Biotechnology 2024, "A community effort to optimize sequence-based deep learning models of gene regulation"); it was the
strongest of the three DREAM-optimised models in that paper, and most clearly so on the larger datasets. The blocks are the published ones from
``benchmarks/human/prixfixe`` of de-Boer-Lab/random-promoter-dream-challenge-2022, composed exactly as that repository's human notebook composes them:
BHIFirstLayersBlock -> BHICoreBlock -> AutosomeFinalLayersBlock.

It is here because every measurement we have says the loop's members are LegNet-derived: a bidirectional LSTM over the sequence is a genuinely different inductive bias from LegNet's convolutional tower, which is
what an ensemble needs if architecture is to be marginalised out rather than merely varied.

Three deviations from the published composition, all forced by our harness and none structural:
  * ``in_channels=4``. The DREAM human pipeline feeds five channels (the four bases plus one flag channel); our harness feeds a 4-channel one-hot. Both first-layer blocks take ``in_channels`` as a parameter, so
    this changes a number and not the graph.
  * ``seqsize=240``, the fixed input standard, in place of their 230. Every block here is length-agnostic (``padding='same'``, pool size 1, and an adaptive average pool in the final block), so the length enters
    only as the bookkeeping argument the blocks carry for compatibility.
  * A single output. The final block already predicts one value, so nothing changes; it is stated because Malinois does need a change here.

The recipe is NOT fixed here. Unlike a proposal, a published reference gets its own per-size tuning: a recipe tuned for LegNet evaluates this architecture at LegNet's optimum, which would understate it exactly
the way our non-LegNet members are already understated.
"""

SOURCE = '''import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """BHI basic convolutional block: conv, max pool, dropout."""

    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.mp = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.mp(x)
        x = self.do(x)
        return x


class BHIFirstLayersBlock(nn.Module):
    """Several ConvBlocks of different kernel sizes, concatenated along the channel dimension."""

    def __init__(self, in_channels=4, out_channels=320, kernel_sizes=(9, 15), pool_size=1, dropout=0.2):
        super().__init__()
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
        each = out_channels // len(kernel_sizes)
        self.out_channels = out_channels
        self.conv_list = nn.ModuleList([ConvBlock(in_channels, each, k, pool_size, dropout) for k in kernel_sizes])

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.conv_list], dim=1)


class BHICoreBlock(nn.Module):
    """A bidirectional LSTM for long-range dependencies, then ConvBlocks that consolidate them, then dropout."""

    def __init__(self, in_channels=320, out_channels=320, lstm_hidden_channels=320,
                 kernel_sizes=(9, 15), pool_size=1, dropout1=0.2, dropout2=0.5):
        super().__init__()
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
        each = out_channels // len(kernel_sizes)
        self.out_channels = out_channels
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_channels, batch_first=True, bidirectional=True)
        self.conv_list = nn.ModuleList([ConvBlock(2 * lstm_hidden_channels, each, k, pool_size, dropout1) for k in kernel_sizes])
        self.do = nn.Dropout(dropout2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = torch.cat([conv(x) for conv in self.conv_list], dim=1)
        return self.do(x)


class AutosomeFinalLayersBlock(nn.Module):
    """Pointwise conv to 256 channels, global average pool, one linear. Length-agnostic."""

    def __init__(self, in_channels=320):
        super().__init__()
        self.mapper = nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1, padding='same')
        self.linear = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x):
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2)
        return self.linear(x)


class Net(nn.Module):
    """DREAM-RNN in the human-benchmark configuration (first 320 ks 9/15 dropout .2, BiLSTM hidden 320, core 320 ks 9/15, dropout .2/.5), on 4-channel one-hot input of any length."""

    def __init__(self):
        super().__init__()
        self.first = BHIFirstLayersBlock(in_channels=4, out_channels=320, kernel_sizes=(9, 15), pool_size=1, dropout=0.2)
        self.core = BHICoreBlock(in_channels=self.first.out_channels, out_channels=320, lstm_hidden_channels=320,
                                 kernel_sizes=(9, 15), pool_size=1, dropout1=0.2, dropout2=0.5)
        self.final = AutosomeFinalLayersBlock(in_channels=self.core.out_channels)

    def forward(self, x):
        x = self.first(x)
        x = self.core(x)
        x = self.final(x)
        return x.squeeze(-1)
'''
