# seeding, accuracy eval and checkpoint loading
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(checkpoint, device, num_classes=10, dropout=0.2):
    # handles both a plain state_dict and a pruned checkpoint (dict with hidden_dims)
    from mobilenetv2 import mobilenet_v2
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model = mobilenet_v2(num_classes=num_classes, dropout=dropout,
                             hidden_dims=ckpt.get('hidden_dims'))
        model.load_state_dict(ckpt['state_dict'])
    else:
        model = mobilenet_v2(num_classes=num_classes, dropout=dropout)
        model.load_state_dict(ckpt)
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, pred = outputs.cpu().max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return 100.0 * correct / total
