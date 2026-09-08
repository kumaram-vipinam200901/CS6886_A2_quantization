# same compression as compress.py but on the pretrained torchvision model
import argparse

import torch

from pretrained_model import build_pretrained_cifar
from compress import run_compression, build_cfg


def _load_pretrained(checkpoint, device, dropout):
    model = build_pretrained_cifar(num_classes=10, pretrained=False, dropout=dropout)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='checkpoints/mobilenetv2_pretrained.pth')
    p.add_argument('--method', choices=['ptq', 'qat'], default='ptq')
    p.add_argument('--w-bits', type=int, default=8)
    p.add_argument('--a-bits', type=int, default=8)
    p.add_argument('--mixed', action='store_true')
    p.add_argument('--target-avg-bits', type=float, default=4.0)
    p.add_argument('--huffman', action='store_true')
    p.add_argument('--w-clip', type=float, default=None)
    p.add_argument('--bias-bits', type=int, default=None)
    p.add_argument('--distill', action='store_true')
    p.add_argument('--teacher-checkpoint', type=str, default=None)
    p.add_argument('--qat-epochs', type=int, default=5)
    p.add_argument('--qat-lr', type=float, default=5e-4)
    p.add_argument('--calib-batches', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--project', type=str, default='cs6886-a2-pretrained')
    p.add_argument('--entity', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = build_cfg(args)

    model = _load_pretrained(cfg['checkpoint'], device, cfg['dropout'])
    teacher = None
    if cfg.get('distill'):
        teacher = _load_pretrained(cfg.get('teacher_checkpoint') or cfg['checkpoint'],
                                   device, cfg['dropout'])

    log_fn = None
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.project, entity=args.entity, config=cfg)
        log_fn = run.log

    run_compression(cfg, log_fn=log_fn, model=model, teacher=teacher)
    if run:
        run.finish()


if __name__ == '__main__':
    main()
