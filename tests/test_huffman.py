"""Huffman coder must be lossless and should beat fixed-width on skewed data."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huffman import build_codes, encode, decode, encoded_bits


def test_roundtrip():
    rng = np.random.default_rng(0)
    # skewed distribution around 0, like quantized weights
    symbols = rng.integers(-7, 8, size=5000)
    symbols[rng.random(5000) < 0.6] = 0
    symbols = symbols.tolist()

    codes = build_codes(symbols)
    data, nbits = encode(symbols, codes)
    assert decode(data, nbits, codes) == symbols


def test_single_symbol():
    symbols = [3] * 100
    codes = build_codes(symbols)
    data, nbits = encode(symbols, codes)
    assert decode(data, nbits, codes) == symbols


def test_beats_fixed_width():
    rng = np.random.default_rng(1)
    symbols = rng.integers(-7, 8, size=5000)
    symbols[rng.random(5000) < 0.7] = 0
    symbols = symbols.tolist()
    fixed = len(symbols) * 4  # 4-bit fixed width for 16 levels
    assert encoded_bits(symbols) < fixed


if __name__ == '__main__':
    test_roundtrip()
    test_single_symbol()
    test_beats_fixed_width()
    print("huffman tests OK")
