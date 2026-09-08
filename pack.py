# write the compressed model to a real .bin (huffman weight codes + fp16 scales +
# int8 biases + activation scale/zero-point) and load it back, so the reported
# size is an actual file and the decoded model can be re-evaluated.
import struct

import numpy as np
import torch

from quant import (named_quant_layers, QuantAct, quantize_weight_per_channel)
from huffman import build_codes, encode, decode

MAGIC = b'CS6A2\x00'


def _bits_to_bytes(bits):
    bits = bits + '0' * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def write_package(model, path):
    buf = bytearray(MAGIC)
    acts = [m for m in model.modules() if isinstance(m, QuantAct)]
    layers = list(named_quant_layers(model))
    buf += struct.pack('<B', acts[0].n_bits if acts else 8)
    buf += struct.pack('<H', len(layers))

    for _, m in layers:
        w = m.weight.detach().cpu()
        q, scale = quantize_weight_per_channel(w, m.n_bits, getattr(m, 'w_clip', None))
        shape = list(w.shape)
        buf += struct.pack('<B', len(shape))
        for s in shape:
            buf += struct.pack('<i', s)
        buf += struct.pack('<B', m.n_bits)
        buf += struct.pack('<B', 1 if m.bias is not None else 0)
        buf += scale.reshape(-1).numpy().astype(np.float16).tobytes()

        if m.bias is not None:
            b = m.bias.detach().cpu()
            bs = float((b.abs().max() / 127).clamp_min(1e-12))
            qb = torch.round(b / bs).clamp(-127, 127).to(torch.int8)
            buf += struct.pack('<e', bs)
            buf += struct.pack('<i', b.numel())
            buf += qb.numpy().tobytes()

        codes = q.reshape(-1).to(torch.int64).numpy().tolist()
        cb = build_codes(codes)
        buf += struct.pack('<H', len(cb))
        for sym, bits in cb.items():
            buf += struct.pack('<h', int(sym))
            buf += struct.pack('<B', len(bits))
            buf += _bits_to_bytes(bits)
        data, nbits = encode(codes, cb)
        buf += struct.pack('<I', nbits)
        buf += data

    buf += struct.pack('<H', len(acts))
    for m in acts:
        buf += struct.pack('<e', float(m.scale))
        buf += struct.pack('<e', float(m.zero_point))

    with open(path, 'wb') as f:
        f.write(buf)
    return len(buf)


@torch.no_grad()
def unpack_into(model, path):
    """Read the .bin and load the decoded weights / biases / activation params
    into `model` (in place). Layers are matched by traversal order."""
    data = open(path, 'rb').read()
    assert data[:6] == MAGIC, 'bad file'
    off = [6]

    def rd(fmt):
        v = struct.unpack_from(fmt, data, off[0])
        off[0] += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    a_bits = rd('<B')
    n_layers = rd('<H')
    layers = list(named_quant_layers(model))
    assert len(layers) == n_layers, 'model does not match the package'

    for _, m in layers:
        ndim = rd('<B')
        shape = [rd('<i') for _ in range(ndim)]
        n_bits = rd('<B')
        has_bias = rd('<B')
        out_ch = shape[0]
        scales = np.frombuffer(data, np.float16, out_ch, off[0]).astype(np.float32); off[0] += out_ch * 2

        bias = None
        if has_bias:
            bs = rd('<e'); bn = rd('<i')
            qb = np.frombuffer(data, np.int8, bn, off[0]).astype(np.float32); off[0] += bn
            bias = torch.tensor(qb * bs)

        n_sym = rd('<H')
        cb = {}
        for _ in range(n_sym):
            sym = rd('<h'); clen = rd('<B')
            nbytes = (clen + 7) // 8
            chunk = data[off[0]:off[0] + nbytes]; off[0] += nbytes
            bits = ''.join(f'{byte:08b}' for byte in chunk)[:clen]
            cb[sym] = bits
        nbits = rd('<I')
        pbytes = (nbits + 7) // 8
        payload = data[off[0]:off[0] + pbytes]; off[0] += pbytes

        codes = decode(payload, nbits, cb)
        q = np.array(codes, np.float32).reshape(shape)
        sc = scales.reshape([out_ch] + [1] * (ndim - 1))
        m.weight.data.copy_(torch.tensor(q * sc))
        m.n_bits = n_bits
        m.quantize = False  # weight is already the dequantized value
        if bias is not None:
            m.bias.data.copy_(bias)

    n_acts = rd('<H')
    for m in [mm for mm in model.modules() if isinstance(mm, QuantAct)]:
        m.n_bits = a_bits
        m.scale.fill_(rd('<e'))
        m.zero_point.fill_(rd('<e'))
    return model
