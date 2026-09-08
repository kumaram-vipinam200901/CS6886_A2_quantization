# main driver: quantize/compress a trained model and print the compression numbers
import argparse

import torch

from data import get_loaders, get_calibration_loader
from utils import set_seed, evaluate, load_model
from quant import convert_to_quant, apply_bit_config, calibrate, named_quant_layers
from qat import train_qat
from sensitivity import measure_sensitivity, layer_sizes, allocate_bits
from size import summarize, print_summary


def run_compression(cfg, log_fn=None, model=None, teacher=None):
    """Compress a model and report accuracy + compression ratios.

    By default the model (and optional distillation teacher) are loaded from
    cfg['checkpoint']. Callers can also pass a prebuilt `model`/`teacher` in,
    which is how the separate pretrained pipeline reuses this same engine.
    """
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_loaders(cfg['batch_size'], cfg['workers'])
    calib_loader = get_calibration_loader(batch_size=64, num_images=512)

    # fp32 baseline accuracy (handles pruned checkpoints too)
    if model is None:
        model = load_model(cfg['checkpoint'], device, dropout=cfg['dropout'])
    model.to(device)
    fp32_acc = evaluate(model, test_loader, device)
    print(f"FP32 accuracy: {fp32_acc:.2f}%")

    # optional FP32 teacher for knowledge distillation
    if cfg.get('distill') and teacher is None:
        teacher = load_model(cfg.get('teacher_checkpoint') or cfg['checkpoint'],
                             device, dropout=cfg['dropout'])
    if teacher is not None:
        teacher.eval()

    # convert to quant modules and calibrate activations
    convert_to_quant(model, w_bits=cfg['w_bits'], a_bits=cfg['a_bits'],
                     w_clip=cfg.get('w_clip'), bias_bits=cfg.get('bias_bits'))
    model.to(device)
    calibrate(model, calib_loader, device, num_batches=cfg['calib_batches'])

    bit_config = None
    if cfg['mixed']:
        # small loader for the sensitivity sweep (fast)
        sens, base_acc = measure_sensitivity(model, calib_loader, device,
                                             low_bits=2, base_bits=cfg['w_bits'])
        sizes = layer_sizes(model)
        choices = tuple(cfg.get('choices', (2, 3, 4, 6, 8)))
        bit_config = allocate_bits(sens, sizes, choices=choices,
                                   target_avg=cfg['target_avg_bits'])
        apply_bit_config(model, bit_config)
        avg = sum(sizes[n] * bit_config[n] for n in bit_config) / sum(sizes.values())
        print(f"Mixed precision: target={cfg['target_avg_bits']} bits, "
              f"achieved avg={avg:.2f} bits over {len(bit_config)} layers")

    if cfg['method'] == 'qat':
        model, _ = train_qat(model, train_loader, test_loader, device,
                             epochs=cfg['qat_epochs'], lr=cfg['qat_lr'],
                             log_fn=log_fn, teacher=teacher)

    quant_acc = evaluate(model, test_loader, device)
    print(f"Quantized accuracy ({cfg['method']}): {quant_acc:.2f}%")

    # activation size uses a single image (per-inference footprint)
    sample = next(iter(test_loader))[0][:1].to(device)
    stats = summarize(model, sample, use_huffman=cfg['huffman'])
    stats['fp32_acc'] = fp32_acc
    stats['quant_acc'] = quant_acc
    stats['acc_drop'] = fp32_acc - quant_acc
    stats['a_bits'] = cfg['a_bits']

    # size-weighted average weight bit-width (equals w_bits when not mixed)
    sizes = {n: m.weight.numel() for n, m in named_quant_layers(model)}
    bits = {n: m.n_bits for n, m in named_quant_layers(model)}
    stats['avg_weight_bits'] = sum(sizes[n] * bits[n] for n in sizes) / sum(sizes.values())

    print_summary(stats)

    # optionally write a real compressed .bin, reload it and re-check accuracy
    if cfg.get('pack'):
        import copy
        from pack import write_package, unpack_into
        nbytes = write_package(model, cfg['pack'])
        m2 = copy.deepcopy(model)
        unpack_into(m2, cfg['pack'])
        acc2 = evaluate(m2, test_loader, device)
        print(f"Packed file on disk: {nbytes / 1024 / 1024:.2f} MB  |  "
              f"decoded model accuracy: {acc2:.2f}%")
        stats['packed_MB'] = nbytes / 1024 / 1024
        stats['decoded_acc'] = acc2

    if log_fn:
        log_fn({k: v for k, v in stats.items() if isinstance(v, (int, float))})
    return stats


def build_cfg(args):
    return {
        'checkpoint': args.checkpoint,
        'method': args.method,
        'w_bits': args.w_bits,
        'a_bits': args.a_bits,
        'mixed': args.mixed,
        'target_avg_bits': args.target_avg_bits,
        'huffman': args.huffman,
        'w_clip': args.w_clip,
        'bias_bits': args.bias_bits,
        'distill': args.distill,
        'teacher_checkpoint': args.teacher_checkpoint,
        'pack': args.pack,
        'qat_epochs': args.qat_epochs,
        'qat_lr': args.qat_lr,
        'calib_batches': args.calib_batches,
        'batch_size': args.batch_size,
        'dropout': args.dropout,
        'workers': args.workers,
        'seed': args.seed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='checkpoints/mobilenetv2_cifar10.pth')
    p.add_argument('--method', choices=['ptq', 'qat'], default='ptq')
    p.add_argument('--w-bits', type=int, default=8)
    p.add_argument('--a-bits', type=int, default=8)
    p.add_argument('--mixed', action='store_true', help='use mixed-precision weights')
    p.add_argument('--target-avg-bits', type=float, default=4.0)
    p.add_argument('--huffman', action='store_true', help='huffman-code the weights')
    p.add_argument('--w-clip', type=float, default=None,
                   help='percentile in (0,1] for weight clipping, e.g. 0.999')
    p.add_argument('--bias-bits', type=int, default=None,
                   help='quantize biases to this many bits (default: keep fp16)')
    p.add_argument('--distill', action='store_true',
                   help='knowledge distillation from an FP32 teacher during QAT')
    p.add_argument('--teacher-checkpoint', type=str, default=None,
                   help='FP32 teacher checkpoint (default: same as --checkpoint)')
    p.add_argument('--pack', type=str, default=None,
                   help='write the compressed model to this .bin, reload and re-check accuracy')
    p.add_argument('--qat-epochs', type=int, default=5)
    p.add_argument('--qat-lr', type=float, default=5e-4)
    p.add_argument('--calib-batches', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--project', type=str, default='cs6886-a2')
    p.add_argument('--entity', type=str, default=None)
    args = p.parse_args()

    log_fn = None
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.project, entity=args.entity, config=build_cfg(args))
        log_fn = run.log

    run_compression(build_cfg(args), log_fn=log_fn)
    if run:
        run.finish()


if __name__ == '__main__':
    main()
