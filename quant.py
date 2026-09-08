# Quantization written from scratch: symmetric per-channel weights, asymmetric
# per-tensor activations (after ReLU6), with BN folded into the conv first.
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_weight_per_channel(w, n_bits, clip=None):
    """Symmetric per-output-channel weight quantization.

    Returns (q, scale) with w ~= q * scale. If clip in (0,1], the range is the
    clip-quantile of |w| instead of the max, which helps at low bit-widths.
    """
    qmax = (1 << (n_bits - 1)) - 1
    w_flat = w.reshape(w.shape[0], -1)
    if clip is not None and clip < 1.0:
        thresh = torch.quantile(w_flat.abs(), clip, dim=1).clamp_min(1e-8)
    else:
        thresh = w_flat.abs().amax(dim=1).clamp_min(1e-8)
    scale = thresh / qmax
    shape = [-1] + [1] * (w.dim() - 1)
    scale = scale.reshape(shape)
    q = torch.round(w / scale).clamp(-qmax, qmax)
    return q, scale


def dequantize_weight(q, scale):
    return q * scale


def quantize_act(x, scale, zero_point, n_bits):
    """Asymmetric unsigned quantization for activations."""
    qmax = (1 << n_bits) - 1
    q = torch.round(x / scale + zero_point).clamp(0, qmax)
    return q


def dequantize_act(q, scale, zero_point):
    return (q - zero_point) * scale


def act_qparams(x_min, x_max, n_bits):
    """Scale / zero-point for an unsigned asymmetric activation range."""
    qmax = (1 << n_bits) - 1
    x_min = min(0.0, float(x_min))  # relu6 output starts at 0
    x_max = max(x_min + 1e-8, float(x_max))
    scale = (x_max - x_min) / qmax
    zero_point = round(-x_min / scale)
    zero_point = min(qmax, max(0, zero_point))
    return scale, zero_point


# BatchNorm folding
def fold_conv_bn(conv, bn):
    """Fold a BatchNorm into the preceding conv, returning a new Conv2d w/ bias."""
    w = conv.weight.clone()
    if conv.bias is not None:
        b = conv.bias.clone()
    else:
        b = torch.zeros(conv.out_channels, device=w.device)

    rstd = (bn.running_var + bn.eps).rsqrt()
    gamma = bn.weight
    beta = bn.bias

    w = w * (gamma * rstd).reshape(-1, 1, 1, 1)
    b = (b - bn.running_mean) * rstd * gamma + beta

    folded = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                       stride=conv.stride, padding=conv.padding,
                       dilation=conv.dilation, groups=conv.groups, bias=True)
    folded.weight.data.copy_(w)
    folded.bias.data.copy_(b)
    return folded


def fold_all_bn(model):
    """Walk every Sequential and fuse Conv2d+BatchNorm2d pairs in place."""
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        i = 0
        while i < len(module) - 1:
            a, b = module[i], module[i + 1]
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                module[i] = fold_conv_bn(a, b)
                module[i + 1] = nn.Identity()
                i += 2
            else:
                i += 1
    return model


# Quantized layers
def _fake_quant_bias(b, bias_bits):
    """Per-tensor symmetric fake-quant of a bias vector (STE)."""
    qmax = (1 << (bias_bits - 1)) - 1
    scale = b.abs().max().clamp_min(1e-12) / qmax
    q = torch.round(b / scale).clamp(-qmax, qmax)
    b_dq = q * scale
    return b + (b_dq - b).detach()


class QuantConv2d(nn.Module):
    """Conv2d whose weights are fake-quantized per output channel.

    Uses the straight-through estimator so it can also be trained (QAT).
    w_clip enables percentile weight clipping; bias_bits (if set) also quantizes
    the bias to that many bits (otherwise the bias is kept in fp16).
    """
    def __init__(self, conv, n_bits=8, w_clip=None, bias_bits=None):
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.n_bits = n_bits
        self.w_clip = w_clip
        self.bias_bits = bias_bits
        self.quantize = True  # set False after unpacking (weight already dequantized)
        self.weight = nn.Parameter(conv.weight.data.clone())
        self.bias = nn.Parameter(conv.bias.data.clone()) if conv.bias is not None else None

    def _fake_quant_weight(self):
        if not self.quantize:
            return self.weight
        q, scale = quantize_weight_per_channel(self.weight, self.n_bits, self.w_clip)
        w_dq = dequantize_weight(q, scale)
        # straight-through estimator: forward uses w_dq, backward flows to weight
        return self.weight + (w_dq - self.weight).detach()

    def _maybe_quant_bias(self):
        if self.bias is None or self.bias_bits is None or not self.quantize:
            return self.bias
        return _fake_quant_bias(self.bias, self.bias_bits)

    def forward(self, x):
        w = self._fake_quant_weight()
        return F.conv2d(x, w, self._maybe_quant_bias(), self.stride, self.padding,
                        self.dilation, self.groups)


class QuantLinear(nn.Module):
    def __init__(self, linear, n_bits=8, w_clip=None, bias_bits=None):
        super().__init__()
        self.n_bits = n_bits
        self.w_clip = w_clip
        self.bias_bits = bias_bits
        self.quantize = True
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

    def _fake_quant_weight(self):
        if not self.quantize:
            return self.weight
        q, scale = quantize_weight_per_channel(self.weight, self.n_bits, self.w_clip)
        w_dq = dequantize_weight(q, scale)
        return self.weight + (w_dq - self.weight).detach()

    def _maybe_quant_bias(self):
        if self.bias is None or self.bias_bits is None or not self.quantize:
            return self.bias
        return _fake_quant_bias(self.bias, self.bias_bits)

    def forward(self, x):
        return F.linear(x, self._fake_quant_weight(), self._maybe_quant_bias())


class QuantAct(nn.Module):
    """Fake-quant for activations. Collects the range during calibration, then
    freezes scale / zero-point and fake-quantizes on every forward."""
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.calibrating = False
        self.enabled = True
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))

    def freeze(self):
        s, zp = act_qparams(self.running_min.item(), self.running_max.item(), self.n_bits)
        self.scale.fill_(s)
        self.zero_point.fill_(float(zp))

    def forward(self, x):
        if self.calibrating:
            self.running_min = torch.minimum(self.running_min, x.min())
            self.running_max = torch.maximum(self.running_max, x.max())
            return x
        if not self.enabled:
            return x
        q = quantize_act(x, self.scale, self.zero_point, self.n_bits)
        x_dq = dequantize_act(q, self.scale, self.zero_point)
        return x + (x_dq - x).detach()  # STE, so QAT can train through it


# Replace conv/linear/relu6 with their quantized versions
def _convert(module, w_bits, a_bits, w_clip, bias_bits):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2d(child, w_bits, w_clip, bias_bits))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear(child, w_bits, w_clip, bias_bits))
        elif isinstance(child, nn.ReLU6):
            setattr(module, name, nn.Sequential(child, QuantAct(a_bits)))
        else:
            _convert(child, w_bits, a_bits, w_clip, bias_bits)


def convert_to_quant(model, w_bits=8, a_bits=8, w_clip=None, bias_bits=None):
    """Fold BN, then replace convs/linears/relu6 with quantized versions.

    Every weight layer starts at `w_bits`; mixed precision is applied afterwards
    with apply_bit_config(). Activation quantizers start at `a_bits`.
    w_clip enables percentile weight clipping; bias_bits quantizes biases.
    """
    fold_all_bn(model)
    _convert(model, w_bits, a_bits, w_clip, bias_bits)
    return model


def named_quant_layers(model):
    """Yield (name, module) for every quantizable weight layer."""
    for name, m in model.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            yield name, m


def apply_bit_config(model, bit_config):
    """Set per-layer weight bits from a {layer_name: n_bits} dict."""
    for name, m in named_quant_layers(model):
        if name in bit_config:
            m.n_bits = int(bit_config[name])


@torch.no_grad()
def calibrate(model, loader, device, num_batches=8):
    """Estimate activation ranges on a few batches, then freeze them."""
    for m in model.modules():
        if isinstance(m, QuantAct):
            m.calibrating = True
    model.eval()
    for i, (x, _) in enumerate(loader):
        model(x.to(device))
        if i + 1 >= num_batches:
            break
    for m in model.modules():
        if isinstance(m, QuantAct):
            m.calibrating = False
            m.freeze()
