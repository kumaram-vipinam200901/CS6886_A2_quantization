# standard MobileNet-v2 but with the early strides set to 1 for 32x32 cifar inputs
import torch
import torch.nn as nn


def _make_divisible(v, divisor=8):
    # keep channel counts divisible by 8 (same trick the original paper uses)
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, hidden_dim=None):
        super().__init__()
        self.stride = stride
        # hidden_dim can be overridden (used by channel pruning); otherwise it is
        # the usual expand_ratio * input channels.
        if hidden_dim is None:
            hidden_dim = int(round(inp * expand_ratio))
        self.hidden_dim = hidden_dim
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pointwise expansion
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers += [
            # depthwise
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pointwise projection (linear, no activation)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.2, hidden_dims=None):
        super().__init__()
        input_channel = 32
        last_channel = 1280

        # t, c, n, s  (expand ratio, out channels, num blocks, stride of first block)
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],   # ImageNet uses stride 2 here; kept 1 for 32x32
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult))

        # first layer: stride 1 for CIFAR (ImageNet uses stride 2)
        features = [ConvBNReLU(3, input_channel, stride=1)]
        used_hidden = []  # record the hidden width actually used per block
        block_idx = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                hd = hidden_dims[block_idx] if hidden_dims is not None else None
                block = InvertedResidual(input_channel, output_channel, stride, t, hidden_dim=hd)
                features.append(block)
                used_hidden.append(block.hidden_dim)
                input_channel = output_channel
                block_idx += 1
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.hidden_dims = used_hidden

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # global average pool
        x = self.classifier(x)
        return x


def mobilenet_v2(num_classes=10, width_mult=1.0, dropout=0.2, hidden_dims=None):
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult,
                       dropout=dropout, hidden_dims=hidden_dims)
