# Model-size accounting and compression ratios. Counts the quantized indices
# plus all overheads: per-channel scales, biases and the Huffman tables.
import numpy as np
import torch

from quant import QuantConv2d, QuantLinear, QuantAct, quantize_weight_per_channel
from huffman import encoded_bits

# storage choices for overheads (bytes)
SCALE_BYTES = 2   # per-channel weight scale stored in fp16
BIAS_BYTES = 2    # folded bias stored in fp16
CODELEN_BITS = 5  # bits to store a Huffman code length in the table


def fp32_size_bytes(model):
    return sum(p.numel() for p in model.parameters()) * 4


@torch.no_grad()
def weight_size_bytes(model, use_huffman=True):
    """Return (compressed_bytes, breakdown_dict) for all quantized weights."""
    index_bytes = 0.0
    scale_bytes = 0.0
    bias_bytes = 0.0
    table_bytes = 0.0

    for _, m in model.named_modules():
        if not isinstance(m, (QuantConv2d, QuantLinear)):
            continue
        clip = getattr(m, 'w_clip', None)
        q, _ = quantize_weight_per_channel(m.weight.detach().cpu(), m.n_bits, clip)
        codes = q.reshape(-1).to(torch.int64).numpy()
        numel = codes.size
        out_ch = m.weight.shape[0]

        if use_huffman:
            payload_bits = encoded_bits(codes.tolist())
            distinct = int(np.unique(codes).size)
            table_bytes += distinct * (m.n_bits + CODELEN_BITS) / 8.0
        else:
            payload_bits = numel * m.n_bits
        index_bytes += payload_bits / 8.0

        scale_bytes += out_ch * SCALE_BYTES
        if m.bias is not None:
            # int8 bias -> 1 byte + one fp16 per-tensor scale; else fp16 each
            bias_bits = getattr(m, 'bias_bits', None)
            if bias_bits is not None:
                bias_bytes += m.bias.numel() * (bias_bits / 8.0) + SCALE_BYTES
            else:
                bias_bytes += m.bias.numel() * BIAS_BYTES

    total = index_bytes + scale_bytes + bias_bytes + table_bytes
    breakdown = {
        'weight_index_bytes': index_bytes,
        'weight_scale_bytes': scale_bytes,
        'weight_table_bytes': table_bytes,
        'bias_bytes': bias_bytes,
        'weight_total_bytes': total,
    }
    return total, breakdown


@torch.no_grad()
def measure_activations(model, sample_input):
    """Run one forward pass and total up activation storage before/after quant.

    Activation compression is measured per inference (at the batch size of
    `sample_input`): FP32 counts every activation element at 32 bits, the
    quantized version counts a_bits per element plus one scale/zero-point pair
    per activation tensor.
    """
    hooks = []
    elems = []
    bits_each = []

    def make_hook(module):
        def hook(mod, inp, out):
            x = inp[0]
            elems.append(x.numel())
            bits_each.append(mod.n_bits)
        return hook

    for m in model.modules():
        if isinstance(m, QuantAct):
            hooks.append(m.register_forward_hook(make_hook(m)))

    was_training = model.training
    model.eval()
    model(sample_input)
    if was_training:
        model.train()
    for h in hooks:
        h.remove()

    total_elems = sum(elems)
    fp32_bits = total_elems * 32
    quant_bits = sum(e * b for e, b in zip(elems, bits_each))
    scale_overhead_bits = len(elems) * (SCALE_BYTES + 1) * 8  # scale + 1 byte zp per tensor
    quant_bits += scale_overhead_bits

    ratio = fp32_bits / max(quant_bits, 1)
    return {
        'act_fp32_bytes': fp32_bits / 8.0,
        'act_quant_bytes': quant_bits / 8.0,
        'act_compression_ratio': ratio,
        'num_act_tensors': len(elems),
    }


@torch.no_grad()
def summarize(model, sample_input, use_huffman=True):
    """Full compression summary used by compress.py and the sweep."""
    fp32 = fp32_size_bytes(model)
    w_bytes, w_break = weight_size_bytes(model, use_huffman=use_huffman)
    acts = measure_activations(model, sample_input)

    fp32_weight_bytes = sum(
        p.numel() for n, p in model.named_parameters() if 'weight' in n
    ) * 4
    # weight-only compressed bytes = codes + scales + huffman tables (no bias)
    weight_only_bytes = (w_break['weight_index_bytes']
                         + w_break['weight_scale_bytes']
                         + w_break['weight_table_bytes'])

    out = {
        'fp32_bytes': fp32,
        'fp32_MB': fp32 / 1024 / 1024,
        'final_model_MB': w_bytes / 1024 / 1024,
        'model_compression_ratio': fp32 / max(w_bytes, 1),
        'weight_compression_ratio': fp32_weight_bytes / max(weight_only_bytes, 1),
    }
    out.update(w_break)
    out.update(acts)
    return out


def print_summary(s):
    print("=== Compression summary ===")
    print(f"FP32 model size            : {s['fp32_MB']:.2f} MB")
    print(f"Final compressed size      : {s['final_model_MB']:.2f} MB")
    print(f"Model compression ratio    : {s['model_compression_ratio']:.2f}x")
    print(f"Weight compression ratio   : {s['weight_compression_ratio']:.2f}x")
    print(f"Activation compression ratio: {s['act_compression_ratio']:.2f}x")
    print("  overheads (bytes): "
          f"scales={s['weight_scale_bytes']:.0f}, "
          f"tables={s['weight_table_bytes']:.0f}, "
          f"bias={s['bias_bytes']:.0f}")
    print("===========================")
