"""BassetBranched model — standalone copy from boda2 (Gosai & Castro, 2025).

Only the layers needed for BassetBranched are included: Conv1dNorm, LinearNorm,
GroupedLinear, RepeatLayer, BranchedLinear.  The LightningModule base class is
replaced with nn.Module so that PyTorch Lightning is not required.

Original source: boda2-main/boda/model/basset.py, boda2-main/boda/model/custom_layers.py
License: MIT (Copyright 2025 Sagar Gosai, Rodrigo Castro)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ── Custom layers (from boda2 custom_layers.py) ─────────────────────────────


class Conv1dNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        batch_norm=True,
        weight_norm=False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        try:
            return self.bn_layer(self.conv(x))
        except AttributeError:
            return self.conv(x)


class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=True, weight_norm=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_features)

    def forward(self, x):
        try:
            return self.bn_layer(self.linear(x))
        except AttributeError:
            return self.linear(x)


class GroupedLinear(nn.Module):
    def __init__(self, in_group_size, out_group_size, groups):
        super().__init__()
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups
        self.weight = nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias = nn.Parameter(torch.zeros(groups, 1, out_group_size))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        reorg = x.permute(1, 0).reshape(self.groups, self.in_group_size, -1).permute(0, 2, 1)
        hook = torch.bmm(reorg, self.weight) + self.bias
        return hook.permute(0, 2, 1).reshape(self.out_group_size * self.groups, -1).permute(1, 0)


class RepeatLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.repeat(*self.args)


class BranchedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_group_size,
        out_group_size,
        n_branches=1,
        n_layers=1,
        activation="ReLU6",
        dropout_p=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.nonlin = getattr(nn, activation)()
        self.dropout = nn.Dropout(p=dropout_p)
        self.intake = RepeatLayer(1, n_branches)
        cur_size = in_features
        for i in range(n_layers):
            out_size = out_group_size if i + 1 == n_layers else hidden_group_size
            setattr(self, f"branched_layer_{i + 1}", GroupedLinear(cur_size, out_size, n_branches))
            cur_size = hidden_group_size

    def forward(self, x):
        hook = self.intake(x)
        i = -1
        for i in range(self.n_layers - 1):
            hook = getattr(self, f"branched_layer_{i + 1}")(hook)
            hook = self.dropout(self.nonlin(hook))
        hook = getattr(self, f"branched_layer_{i + 2}")(hook)
        return hook


# ── Helper ───────────────────────────────────────────────────────────────────


def _get_padding(kernel_size):
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    return [max(0, x) for x in [left, right]]


# ── BassetBranched ───────────────────────────────────────────────────────────


class BassetBranched(nn.Module):
    """Malinois CNN architecture (Gosai et al. 2024).

    3 conv layers → 2 shared linear layers → branched linear → grouped output.
    Expects input shape (batch, 4, input_len) with input_len=600.
    """

    def __init__(
        self,
        input_len=600,
        conv1_channels=300,
        conv1_kernel_size=19,
        conv2_channels=200,
        conv2_kernel_size=11,
        conv3_channels=200,
        conv3_kernel_size=7,
        n_linear_layers=2,
        linear_channels=1000,
        linear_activation="ReLU",
        linear_dropout_p=0.3,
        n_branched_layers=1,
        branched_channels=250,
        branched_activation="ReLU6",
        branched_dropout_p=0.0,
        n_outputs=1,
        use_batch_norm=True,
        use_weight_norm=False,
    ):
        super().__init__()
        self.input_len = input_len
        self.n_linear_layers = n_linear_layers
        self.n_outputs = n_outputs

        self.pad1 = nn.ConstantPad1d(_get_padding(conv1_kernel_size), 0.0)
        self.conv1 = Conv1dNorm(
            4,
            conv1_channels,
            conv1_kernel_size,
            batch_norm=use_batch_norm,
            weight_norm=use_weight_norm,
        )
        self.pad2 = nn.ConstantPad1d(_get_padding(conv2_kernel_size), 0.0)
        self.conv2 = Conv1dNorm(
            conv1_channels,
            conv2_channels,
            conv2_kernel_size,
            batch_norm=use_batch_norm,
            weight_norm=use_weight_norm,
        )
        self.pad3 = nn.ConstantPad1d(_get_padding(conv3_kernel_size), 0.0)
        self.conv3 = Conv1dNorm(
            conv2_channels,
            conv3_channels,
            conv3_kernel_size,
            batch_norm=use_batch_norm,
            weight_norm=use_weight_norm,
        )
        self.pad4 = nn.ConstantPad1d((1, 1), 0.0)

        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)

        flatten_factor = self._get_flatten_factor(input_len)
        next_in = conv3_channels * flatten_factor

        for i in range(n_linear_layers):
            setattr(
                self,
                f"linear{i + 1}",
                LinearNorm(
                    next_in, linear_channels, batch_norm=use_batch_norm, weight_norm=use_weight_norm
                ),
            )
            next_in = linear_channels

        self.branched = BranchedLinear(
            next_in,
            branched_channels,
            branched_channels,
            n_outputs,
            n_branched_layers,
            branched_activation,
            branched_dropout_p,
        )
        self.output = GroupedLinear(branched_channels, 1, n_outputs)

        self.nonlin = getattr(nn, linear_activation)()
        self.dropout = nn.Dropout(p=linear_dropout_p)

    @staticmethod
    def _get_flatten_factor(input_len):
        hook = input_len
        assert hook % 3 == 0
        hook = hook // 3
        assert hook % 4 == 0
        hook = hook // 4
        assert (hook + 2) % 4 == 0
        return (hook + 2) // 4

    def encode(self, x):
        hook = self.nonlin(self.conv1(self.pad1(x)))
        hook = self.maxpool_3(hook)
        hook = self.nonlin(self.conv2(self.pad2(hook)))
        hook = self.maxpool_4(hook)
        hook = self.nonlin(self.conv3(self.pad3(hook)))
        hook = self.maxpool_4(self.pad4(hook))
        return torch.flatten(hook, start_dim=1)

    def decode(self, x):
        hook = x
        for i in range(self.n_linear_layers):
            hook = self.dropout(self.nonlin(getattr(self, f"linear{i + 1}")(hook)))
        return self.branched(hook)

    def forward(self, x):
        return self.output(self.decode(self.encode(x)))
