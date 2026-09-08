# Simple Huffman coder for the quantized weight indices. Their values cluster
# around zero, so variable-length codes beat a fixed n-bit code.
import heapq
from collections import Counter


def build_codes(symbols):
    """Return {symbol: bitstring} for the given iterable of integer symbols."""
    freq = Counter(symbols)
    if len(freq) == 1:
        # only one distinct value -> give it a 1-bit code
        return {next(iter(freq)): '0'}

    # each heap entry: (weight, tie_breaker, node)
    # node is either a symbol (leaf) or [left, right]
    heap = [(f, i, s) for i, (s, f) in enumerate(freq.items())]
    heapq.heapify(heap)
    counter = len(heap)
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, counter, [n1, n2]))
        counter += 1

    codes = {}

    def walk(node, prefix):
        if isinstance(node, list):
            walk(node[0], prefix + '0')
            walk(node[1], prefix + '1')
        else:
            codes[node] = prefix

    walk(heap[0][2], '')
    return codes


def encode(symbols, codes):
    """Encode symbols to (bytes, num_bits) using the code table."""
    bits = ''.join(codes[s] for s in symbols)
    num_bits = len(bits)
    # pad up to a whole number of bytes
    bits += '0' * (-num_bits % 8)
    data = bytearray(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    return bytes(data), num_bits


def decode(data, num_bits, codes):
    """Inverse of encode(). Returns the list of symbols."""
    inv = {v: k for k, v in codes.items()}
    bits = ''.join(f'{byte:08b}' for byte in data)[:num_bits]
    out = []
    cur = ''
    for b in bits:
        cur += b
        if cur in inv:
            out.append(inv[cur])
            cur = ''
    return out


def encoded_bits(symbols):
    """Payload bits if we Huffman-encode these symbols."""
    codes = build_codes(symbols)
    freq = Counter(symbols)
    return sum(len(codes[s]) * n for s, n in freq.items())
