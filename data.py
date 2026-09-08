# CIFAR-10 loaders. Train uses crop + flip + AutoAugment + random erase; test and
# calibration use only normalization. Calibration = a small train subset used to
# estimate activation ranges.
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# CIFAR-10 channel statistics (0-1 scale)
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2470, 0.2435, 0.2616]


def _train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2)),  # cutout-style
    ])


def _test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def get_loaders(batch_size=128, num_workers=4, root='./data'):
    train_set = torchvision.datasets.CIFAR10(root, train=True, download=True,
                                             transform=_train_transform())
    test_set = torchvision.datasets.CIFAR10(root, train=False, download=True,
                                            transform=_test_transform())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_calibration_loader(batch_size=64, num_images=512, num_workers=0, root='./data'):
    """A small subset of the training set with test transforms (no augmentation).

    num_workers defaults to 0 because this loader is re-iterated many times
    during the sensitivity sweep and worker startup overhead would dominate.
    """
    cal_set = torchvision.datasets.CIFAR10(root, train=True, download=True,
                                           transform=_test_transform())
    idx = torch.randperm(len(cal_set))[:num_images].tolist()
    cal_set = Subset(cal_set, idx)
    return DataLoader(cal_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
