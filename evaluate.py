import argparse
import torch

from data import get_loaders
from utils import evaluate, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='checkpoints/mobilenetv2_cifar10.pth')
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_loaders(args.batch_size)
    model = load_model(args.checkpoint, device)
    print(f"Test accuracy: {evaluate(model, test_loader, device):.2f}%")


if __name__ == '__main__':
    main()
