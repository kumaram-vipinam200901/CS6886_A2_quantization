"""Size accounting sanity: fp32 size is numel*4 and quantizing shrinks it."""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobilenetv2 import mobilenet_v2
from quant import convert_to_quant, calibrate
from size import fp32_size_bytes, weight_size_bytes, measure_activations


def test_fp32_size():
    model = mobilenet_v2(num_classes=10)
    expected = sum(p.numel() for p in model.parameters()) * 4
    assert fp32_size_bytes(model) == expected


def test_quant_smaller():
    model = mobilenet_v2(num_classes=10)
    fp32_w = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n) * 4
    convert_to_quant(model, w_bits=4, a_bits=8)
    comp, _ = weight_size_bytes(model, use_huffman=True)
    assert comp < fp32_w


def test_activation_measure_runs():
    model = mobilenet_v2(num_classes=10)
    convert_to_quant(model, w_bits=8, a_bits=8)
    loader = [(torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))]
    calibrate(model, loader, torch.device('cpu'), num_batches=1)
    stats = measure_activations(model, torch.randn(1, 3, 32, 32))
    assert stats['act_compression_ratio'] > 1.0


if __name__ == '__main__':
    test_fp32_size()
    test_quant_smaller()
    test_activation_measure_runs()
    print("size tests OK")
