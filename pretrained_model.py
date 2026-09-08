# download torchvision's imagenet-pretrained mobilenetv2 and adapt it to cifar-10
# (stem + first stage stride -> 1, new 10-class head)
import torch.nn as nn
import torchvision
from torchvision.models import MobileNet_V2_Weights


def build_pretrained_cifar(num_classes=10, pretrained=True, dropout=0.2):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.mobilenet_v2(weights=weights, dropout=dropout)

    # keep resolution for 32x32 inputs (ImageNet version downsamples 32x)
    model.features[0][0].stride = (1, 1)          # stem conv: stride 2 -> 1
    model.features[2].conv[1][0].stride = (1, 1)  # 2nd stage depthwise: stride 2 -> 1

    # new classifier head for CIFAR-10
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
