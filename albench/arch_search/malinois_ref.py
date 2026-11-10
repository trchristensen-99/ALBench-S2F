"""Malinois written to the AutoResearch contract (a ``Net`` class without constructor arguments), so it trains in the same harness as MPRA-LegNet and every proposal.

Malinois is the Basset-derived CNN from Gosai et al., Nature 2024 ("Machine-guided design of cell-type-targeting cis-regulatory elements"), which predicts MPRA activity in K562, HepG2 and SK-N-SH. The layers are
the published ``Basset`` of sjgosai/boda2 (``boda/model/basset.py`` and ``boda/model/custom_layers.py``) at its own default widths -- conv 300 ks 19, 200 ks 11, 200 ks 7, max pools 3/4/4, two 1000-unit dense layers,
dropout 0.3, batch norm on and weight norm off -- which reproduce the ~4.5M parameters the DREAM benchmark paper reports for Malinois at a 600-bp input, so these are its architecture and not merely Basset's.

It belongs in the bank for a reason the DREAM models do not cover: it is the architecture of the assay our labels descend from, so it is the strongest published CNN prior for *this* readout. That same closeness is
why only the architecture is taken. **The published Malinois weights must never be loaded here.** They were fit on the Gosai measurements our oracle was built from; as weights that is leakage and would make this a
baseline next to the AlphaGenome students, whereas re-implemented and trained from scratch on a training corpus it is an ordinary member, on exactly the footing MPRA-LegNet already has.

Deviations from the published model, all forced by our harness:
  * ``n_outputs=1``. Malinois predicts three cell types; we predict one K562 value.
  * Regression head only. The published class carries a ``CrossEntropyLoss`` default and a Lightning training loop; the harness owns the loss, so neither is reproduced.
  * The ``activation`` string argument is gone and ``nn.ReLU()`` is named directly. The published ``Basset`` resolves it with ``getattr(nn, activation)``, which the harness's source validator rejects
    (``albench/arch_search/validate.py``, ``BANNED_NAMES``). ReLU is the published default and the only value Malinois uses, so this removes a knob rather than changing the model.
  * **The dense width is derived from the input length rather than hard-coded.** ``Basset.linear1`` takes ``conv3_channels * 13``, where 13 is what 600 bp becomes after pools of 3, 4 and 4. Our fixed standard is
    240 bp, which becomes 5, so this model at our input is ~3.0M parameters against the published ~4.5M. The alternative -- padding 240 bp out to 600 -- would break comparability with every other student in the
    study, so the width moves instead. A reader comparing our Malinois numbers with the paper's should know the dense stack is the part that shrank.

The recipe is NOT fixed here: a published reference gets its own per-size tuning, because a LegNet-tuned recipe would evaluate this architecture at another model's optimum.
"""

SOURCE = '''import torch
import torch.nn as nn


def get_padding(kernel_size):
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    return [max(0, x) for x in [left, right]]


class Conv1dNorm(nn.Module):
    """Convolution with optional batch norm and weight norm, as published."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, batch_norm=True, weight_norm=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        self.bn_layer = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn_layer(x) if self.bn_layer is not None else x


class LinearNorm(nn.Module):
    """Linear layer with optional batch norm and weight norm, as published."""

    def __init__(self, in_features, out_features, bias=True, batch_norm=True, weight_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        self.bn_layer = nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if batch_norm else None

    def forward(self, x):
        x = self.linear(x)
        return self.bn_layer(x) if self.bn_layer is not None else x


class Basset(nn.Module):
    """The published Basset encoder/decoder at Malinois's widths, with the dense width derived from the input length."""

    def __init__(self, input_len=240, conv1_channels=300, conv1_kernel_size=19,
                 conv2_channels=200, conv2_kernel_size=11, conv3_channels=200, conv3_kernel_size=7,
                 linear1_channels=1000, linear2_channels=1000, n_outputs=1,
                 dropout_p=0.3, use_batch_norm=True, use_weight_norm=False):
        super().__init__()
        self.pad1 = nn.ConstantPad1d(get_padding(conv1_kernel_size), 0.0)
        self.conv1 = Conv1dNorm(4, conv1_channels, conv1_kernel_size, stride=1, padding=0, dilation=1,
                                groups=1, bias=True, batch_norm=use_batch_norm, weight_norm=use_weight_norm)
        self.pad2 = nn.ConstantPad1d(get_padding(conv2_kernel_size), 0.0)
        self.conv2 = Conv1dNorm(conv1_channels, conv2_channels, conv2_kernel_size, stride=1, padding=0, dilation=1,
                                groups=1, bias=True, batch_norm=use_batch_norm, weight_norm=use_weight_norm)
        self.pad3 = nn.ConstantPad1d(get_padding(conv3_kernel_size), 0.0)
        self.conv3 = Conv1dNorm(conv2_channels, conv3_channels, conv3_kernel_size, stride=1, padding=0, dilation=1,
                                groups=1, bias=True, batch_norm=use_batch_norm, weight_norm=use_weight_norm)
        self.pad4 = nn.ConstantPad1d((1, 1), 0.0)
        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)

        # published code takes an ``activation`` string and resolves it with getattr; the harness's
        # source validator bans getattr, so the default ReLU -- the only value Malinois uses -- is named
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

        # published code hard-codes conv3_channels * 13, which is 600 bp after pools of 3, 4 and 4;
        # measure it instead so the harness's input length is the one that decides
        with torch.no_grad():
            flat = self.encode(torch.zeros(1, 4, input_len)).shape[1]
        self.linear1 = LinearNorm(flat, linear1_channels, bias=True, batch_norm=use_batch_norm, weight_norm=use_weight_norm)
        self.linear2 = LinearNorm(linear1_channels, linear2_channels, bias=True, batch_norm=use_batch_norm, weight_norm=use_weight_norm)
        self.output = nn.Linear(linear2_channels, n_outputs)

    def encode(self, x):
        hook = self.nonlin(self.conv1(self.pad1(x)))
        hook = self.maxpool_3(hook)
        hook = self.nonlin(self.conv2(self.pad2(hook)))
        hook = self.maxpool_4(hook)
        hook = self.nonlin(self.conv3(self.pad3(hook)))
        hook = self.maxpool_4(self.pad4(hook))
        return torch.flatten(hook, start_dim=1)

    def decode(self, x):
        hook = self.dropout(self.nonlin(self.linear1(x)))
        hook = self.dropout(self.nonlin(self.linear2(hook)))
        return hook

    def forward(self, x):
        return self.output(self.decode(self.encode(x)))


class Net(nn.Module):
    """Malinois's architecture (Basset widths 300/200/200, kernels 19/11/7, pools 3/4/4, dense 1000/1000, dropout .3) with a single regression output, trained from scratch -- never with the published weights."""

    def __init__(self):
        super().__init__()
        self.basset = Basset(input_len=240, n_outputs=1)

    def forward(self, x):
        return self.basset(x).squeeze(-1)
'''
