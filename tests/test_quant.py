"""Sanity checks for the quantizers and BN folding."""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant import (quantize_weight_per_channel, dequantize_weight,
                   fold_conv_bn)


def _per_tensor_error(w, n_bits):
    qmax = (1 << (n_bits - 1)) - 1
    scale = w.abs().max() / qmax
    q = torch.round(w / scale).clamp(-qmax, qmax)
    return (q * scale - w).abs().mean().item()


def test_per_channel_beats_per_tensor():
    torch.manual_seed(0)
    # channels with very different scales, like depthwise conv
    w = torch.randn(16, 1, 3, 3)
    w[0] *= 100.0
    w[1] *= 0.01
    q, scale = quantize_weight_per_channel(w, 4)
    pc_err = (dequantize_weight(q, scale) - w).abs().mean().item()
    pt_err = _per_tensor_error(w, 4)
    assert pc_err < pt_err


def test_bn_fold_matches():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    bn = nn.BatchNorm2d(8)
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 1.5)
    bn.weight.data.normal_()
    bn.bias.data.normal_()
    bn.eval()

    x = torch.randn(2, 3, 16, 16)
    ref = bn(conv(x))
    folded = fold_conv_bn(conv, bn)
    out = folded(x)
    assert torch.allclose(ref, out, atol=1e-4)


if __name__ == '__main__':
    test_per_channel_beats_per_tensor()
    test_bn_fold_matches()
    print("quant tests OK")
