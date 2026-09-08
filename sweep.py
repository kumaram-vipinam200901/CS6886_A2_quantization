# run a bunch of configs and log to wandb, then build the parallel coordinates chart
import argparse
import wandb

from compress import run_compression


def build_run_list():
    """The configurations we sweep. Mixes uniform and mixed-precision, PTQ and QAT."""
    runs = []

    # uniform-precision PTQ across weight bit-widths
    for w in [8, 6, 4, 3, 2]:
        runs.append({'method': 'ptq', 'mixed': False, 'w_bits': w,
                     'a_bits': 8, 'target_avg_bits': w})

    # uniform-precision QAT at the aggressive settings
    for w in [4, 3, 2]:
        runs.append({'method': 'qat', 'mixed': False, 'w_bits': w,
                     'a_bits': 6, 'target_avg_bits': w})

    # mixed-precision PTQ and QAT at a few target average bit-widths
    for t in [4, 3, 2]:
        runs.append({'method': 'ptq', 'mixed': True, 'w_bits': 8,
                     'a_bits': 8, 'target_avg_bits': t})
        runs.append({'method': 'qat', 'mixed': True, 'w_bits': 8,
                     'a_bits': 6, 'target_avg_bits': t})
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/mobilenetv2_cifar10.pth')
    p.add_argument('--project', default='cs6886-a2')
    p.add_argument('--entity', default=None)
    p.add_argument('--qat-epochs', type=int, default=3)
    p.add_argument('--count', type=int, default=None, help='limit number of runs')
    args = p.parse_args()

    run_list = build_run_list()
    if args.count:
        run_list = run_list[:args.count]

    for i, r in enumerate(run_list):
        cfg = {
            'checkpoint': args.checkpoint,
            'method': r['method'],
            'w_bits': r['w_bits'],
            'a_bits': r['a_bits'],
            'mixed': r['mixed'],
            'target_avg_bits': r['target_avg_bits'],
            'huffman': True,
            'qat_epochs': args.qat_epochs,
            'qat_lr': 5e-4,
            'calib_batches': 8,
            'batch_size': 128,
            'dropout': 0.2,
            'workers': 4,
            'seed': 42,
        }
        name = f"{r['method']}_{'mixed' if r['mixed'] else 'w' + str(r['w_bits'])}_a{r['a_bits']}"
        print(f"\n=== run {i + 1}/{len(run_list)}: {name} ===")
        run = wandb.init(project=args.project, entity=args.entity, name=name,
                         config=cfg, reinit=True)
        run_compression(cfg, log_fn=run.log)
        run.finish()


if __name__ == '__main__':
    main()
