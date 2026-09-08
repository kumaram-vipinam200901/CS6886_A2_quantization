# train the fp32 baseline on cifar-10
import argparse
import json
import os

import torch
import torch.nn as nn

from mobilenetv2 import mobilenet_v2
from data import get_loaders
from utils import set_seed, evaluate
from plots import plot_history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', type=str, default='checkpoints/mobilenetv2_cifar10.pth')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--project', type=str, default='cs6886-a2')
    p.add_argument('--entity', type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.project, entity=args.entity,
                         name='baseline', config=vars(args))

    train_loader, test_loader = get_loaders(args.batch_size, args.workers)
    model = mobilenet_v2(num_classes=10, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
        scheduler.step()

        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss={train_loss:.1f}  "
              f"train_acc={train_acc:.2f}%  test_acc={test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.out)

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'train_acc': train_acc, 'test_acc': test_acc,
                        'lr': scheduler.get_last_lr()[0]})
        if run:
            run.log(history[-1])

    # save history + plot curves (useful even without wandb)
    os.makedirs('curves', exist_ok=True)
    with open('curves/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    plot_history(history, 'curves')

    print(f"Best test accuracy: {best_acc:.2f}%  (saved to {args.out})")
    print("Curves written to curves/loss.png and curves/accuracy.png")
    if run:
        run.summary['best_test_acc'] = best_acc
        run.finish()


if __name__ == '__main__':
    main()
