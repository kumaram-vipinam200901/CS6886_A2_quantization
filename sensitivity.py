# Mixed precision: measure each layer's sensitivity (accuracy drop when it alone
# is set to 2-bit), then give sensitive layers more bits to hit a target average.
import torch

from quant import named_quant_layers
from utils import evaluate


@torch.no_grad()
def measure_sensitivity(model, loader, device, low_bits=2, base_bits=8):
    """Return {layer_name: accuracy_drop} when that layer alone goes to low_bits."""
    layers = dict(named_quant_layers(model))
    for m in layers.values():
        m.n_bits = base_bits
    base_acc = evaluate(model, loader, device)

    sensitivity = {}
    for name, m in layers.items():
        original = m.n_bits
        m.n_bits = low_bits
        acc = evaluate(model, loader, device)
        sensitivity[name] = base_acc - acc
        m.n_bits = original
    return sensitivity, base_acc


def layer_sizes(model):
    return {name: m.weight.numel() for name, m in named_quant_layers(model)}


def allocate_bits(sensitivity, sizes, choices=(2, 4, 8), target_avg=4.0):
    """Greedy allocation: start everyone high, drop the least sensitive layers
    step by step until the (size-weighted) average bit-width meets the target."""
    choices = sorted(choices)
    bits = {name: choices[-1] for name in sensitivity}
    total_params = sum(sizes.values())

    def avg_bits():
        return sum(sizes[n] * bits[n] for n in bits) / total_params

    order = sorted(sensitivity, key=lambda n: sensitivity[n])  # least sensitive first
    changed = True
    while avg_bits() > target_avg and changed:
        changed = False
        for n in order:
            lowers = [c for c in choices if c < bits[n]]
            if lowers:
                bits[n] = lowers[-1]
                changed = True
                if avg_bits() <= target_avg:
                    return bits
    return bits
