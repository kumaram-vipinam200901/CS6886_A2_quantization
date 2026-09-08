# prune channels then finetune the smaller model (saves hidden dims with the weights)
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_loaders
from utils import set_seed, evaluate, load_model
from pruning import make_pruned_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/mobilenetv2_cifar10.pth')
    p.add_argument('--prune-ratio', type=float, default=0.3)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--distill', action='store_true')
    p.add_argument('--kd-alpha', type=float, default=0.5)
    p.add_argument('--kd-temp', type=float, default=4.0)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='checkpoints/mobilenetv2_pruned.pth')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--project', default='cs6886-a2')
    p.add_argument('--entity', default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.project, entity=args.entity,
                         name=f'prune{args.prune_ratio}', config=vars(args))

    train_loader, test_loader = get_loaders(args.batch_size, args.workers)

    dense = load_model(args.checkpoint, device)
    dense_acc = evaluate(dense, test_loader, device)
    print(f"Dense accuracy: {dense_acc:.2f}%")

    model, dims = make_pruned_model(dense, args.prune_ratio)
    model.to(device)
    print(f"Pruned hidden dims: {dims}")
    print(f"Params: dense={sum(p.numel() for p in dense.parameters()):,} "
          f"pruned={sum(p.numel() for p in model.parameters()):,}")
    print(f"Accuracy right after pruning (no fine-tune): "
          f"{evaluate(model, test_loader, device):.2f}%")

    teacher = dense if args.distill else None
    if teacher is not None:
        teacher.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, targets)
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(inputs)
                kd = F.kl_div(F.log_softmax(out / args.kd_temp, dim=1),
                              F.softmax(t_out / args.kd_temp, dim=1),
                              reduction='batchmean') * (args.kd_temp ** 2)
                loss = args.kd_alpha * kd + (1 - args.kd_alpha) * loss
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss={running:.1f}  test_acc={acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save({'state_dict': model.state_dict(), 'hidden_dims': dims}, args.out)
        if run:
            run.log({'epoch': epoch + 1, 'loss': running, 'test_acc': acc})

    print(f"Best pruned accuracy: {best_acc:.2f}%  (saved to {args.out})")
    if run:
        run.summary['best_test_acc'] = best_acc
        run.finish()


if __name__ == '__main__':
    main()
