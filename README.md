# MobileNet-v2 CIFAR-10 Compression (CS6886 Assignment 2)

Train MobileNet-v2 on CIFAR-10 and compress it with a self-written quantizer
(no quantization/compression libraries used). Supports post-training
quantization (PTQ) and quantization-aware training (QAT), per-channel weight
quantization, mixed-precision bit allocation, and a hand-written Huffman coder.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Environment used: Python 3.11, torch 2.11.0 (CUDA 12.8), torchvision 0.26.0.
All runs are seeded (`--seed 42`, default in every script).

## Dataset and checkpoints

The `data/` and `checkpoints/` folders are **not** in the repo. CIFAR-10 is
downloaded automatically to `./data` the first time you run any script (the
loaders use `download=True`), so no manual step is needed. To fetch it up front:

```bash
python -c "import torchvision; torchvision.datasets.CIFAR10('./data', train=True, download=True); torchvision.datasets.CIFAR10('./data', train=False, download=True)"
```

Checkpoints (`checkpoints/*.pth`) are produced by `train.py` and `prune.py` when
you run the commands below.

## Reproduce the reported result (94.88% -> 93.67% at 18.99x)

Run these three commands in order (drop `--wandb ...` to run without logging):

```bash
# 1. train the FP32 baseline  -> checkpoints/mobilenetv2_cifar10.pth (94.88%)
python train.py --epochs 150 --batch-size 128 --lr 0.05 --seed 42

# 2. prune 30% of the expansion channels and fine-tune -> checkpoints/mobilenetv2_pruned.pth
python prune.py --checkpoint checkpoints/mobilenetv2_cifar10.pth \
    --prune-ratio 0.3 --epochs 40 --distill --out checkpoints/mobilenetv2_pruned.pth

# 3. compress the pruned model (mixed 3-bit weights, 6-bit activations, Huffman)
#    --pack writes the actual compressed .bin, reloads it and re-checks accuracy
python compress.py --checkpoint checkpoints/mobilenetv2_pruned.pth \
    --method qat --mixed --target-avg-bits 3 --a-bits 6 --w-clip 0.999 \
    --bias-bits 8 --distill --qat-epochs 8 --huffman \
    --pack checkpoints/compressed_best.bin
```

Step 3 prints the compression summary and, with `--pack`, writes a real
`checkpoints/compressed_best.bin` (~0.45 MB), reloads it and reports the decoded
model's accuracy (~93.7%), so the size/accuracy are measured, not estimated.

## Results

What I got on CIFAR-10 (all sizes compared against the 8.53 MB FP32 model):

- Baseline (no compression): 94.88% top-1, 8.53 MB.
- 8-bit PTQ: basically lossless (94.79%) at ~4.4x, still 1.92 MB.
- Best config (30% pruning + mixed ~3-bit weights + 6-bit activations + Huffman,
  with distillation): 93.67% at 18.99x smaller, i.e. 0.45 MB. Weights shrink
  19.38x and activations 7.39x.
- Packing that model to a real .bin and loading it back still gives 93.66%, so
  the 0.45 MB is an actual file, not a formula.
- If you care more about accuracy, plain QAT at 4-bit weights / 6-bit activations
  gives 94.41% (only 0.5% below baseline) at ~10x.
- Pretrained torchvision variant (ablation): 96.24% FP32, 93.76% compressed at 13.5x.

The 2-bit runs fall apart (~10-18%), which is why the mixed-precision + pruning
route is the one I went with.

**Note (this machine):** a system-wide `PYTHONPATH` was set that leaks the global
site-packages into any venv. To keep the venv's package versions authoritative,
clear it before running, and (if you want to reuse the already-installed CUDA
build of torch instead of re-downloading ~2.7 GB) create the venv with
`--system-site-packages`:

```powershell
python -m venv .venv --system-site-packages
$env:PYTHONPATH = ""      # run this in each shell before using the venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## 1. Train the baseline

```bash
python train.py --epochs 150 --batch-size 128 --lr 0.05 --seed 42 --wandb
```

Best weights are saved to `checkpoints/mobilenetv2_cifar10.pth`. Evaluate with:

```bash
python evaluate.py --checkpoint checkpoints/mobilenetv2_cifar10.pth
```

## 2. Compress

```bash
# plain 8-bit PTQ
python compress.py --method ptq --w-bits 8 --a-bits 8 --huffman

# mixed-precision PTQ, ~4 average weight bits, 8-bit activations
python compress.py --method ptq --mixed --target-avg-bits 4 --a-bits 8 --huffman

# QAT fine-tune at 4-bit weights / 6-bit activations
python compress.py --method qat --w-bits 4 --a-bits 6 --qat-epochs 5 --huffman

# best config: mixed precision + QAT + distillation + weight clipping + int8 bias
python compress.py --method qat --mixed --target-avg-bits 3 --a-bits 6 \
    --w-clip 0.999 --bias-bits 8 --distill --qat-epochs 5 --huffman
```

`compress.py` prints FP32 accuracy, quantized accuracy, and the model / weight /
activation compression ratios plus the final model size in MB.

Extra options: `--w-clip 0.999` clips weight outliers for a tighter low-bit
range, `--bias-bits 8` stores biases in int8 instead of fp16, and `--distill`
trains the quantized model to match the FP32 model's outputs (knowledge
distillation) to recover accuracy at low bits.

## Alternative: pretrained (torchvision) MobileNet-v2

A separate, parallel pipeline downloads the real ImageNet-pretrained
MobileNet-v2, adapts it to CIFAR-10, fine-tunes it, and compresses it with the
same engine. It logs to its own wandb project (`cs6886-a2-pretrained`) so it
never mixes with the from-scratch results.

```bash
python train_pretrained.py --epochs 30 --lr 0.01 --wandb \
    --project cs6886-a2-pretrained --entity <your-entity>

python compress_pretrained.py --method qat --mixed --target-avg-bits 3 --a-bits 6 \
    --w-clip 0.999 --bias-bits 8 --distill --qat-epochs 8 --huffman --wandb \
    --project cs6886-a2-pretrained --entity <your-entity>
```

The from-scratch model (`train.py` / `compress.py`) is the primary submission;
this variant is an ablation showing the effect of ImageNet initialization.

## Optional: channel pruning first

`prune.py` removes the least important expansion channels from each
inverted-residual block and fine-tunes the smaller model, so pruning and
quantization stack for a higher overall compression ratio.

```bash
python prune.py --checkpoint checkpoints/mobilenetv2_cifar10.pth \
    --prune-ratio 0.3 --epochs 40 --distill --out checkpoints/mobilenetv2_pruned.pth

# then compress the pruned model (same commands, just point at the pruned file)
python compress.py --checkpoint checkpoints/mobilenetv2_pruned.pth \
    --method qat --mixed --target-avg-bits 3 --a-bits 6 --w-clip 0.999 \
    --bias-bits 8 --distill --qat-epochs 5 --huffman
```

## 3. wandb sweep (parallel coordinates chart)

```bash
python sweep.py --checkpoint checkpoints/mobilenetv2_cifar10.pth
# python sweep.py --count 6      # only the first few configs (quick test)
```

This logs a set of PTQ/QAT runs (uniform and mixed-precision) to one wandb
project. In wandb, add a **Parallel Coordinates** panel over `avg_weight_bits`,
`a_bits`, `method`, `model_compression_ratio`, `final_model_MB`, and `quant_acc`.
To run offline: `set WANDB_MODE=offline` (Windows) / `export WANDB_MODE=offline`,
then `wandb sync` later.

## Training curves

`train.py` writes `curves/history.json` plus `curves/loss.png` and
`curves/accuracy.png`. Regenerate the plots any time with:

```bash
python plots.py --history curves/history.json
```

## Method (short)

- **BN folding**: each BatchNorm is folded into the preceding conv before quantizing.
- **Weights**: symmetric, per-output-channel (needed for depthwise convs whose
  channel ranges differ a lot). Configurable bit-width.
- **Activations**: asymmetric, per-tensor, unsigned (they follow ReLU6). Ranges
  are calibrated on a small subset of the training set.
- **Mixed precision**: measure each layer's sensitivity (accuracy drop when
  quantized alone to 2 bits) and give sensitive layers more bits, aiming for a
  target average bit-width.
- **Huffman coding**: quantized weight indices are entropy-coded with a
  from-scratch Huffman coder for extra, lossless compression.
- **Size accounting** (`size.py`) includes per-channel scales, folded biases,
  and Huffman code tables as overhead.

## Tests

```bash
python tests/test_huffman.py
python tests/test_quant.py
python tests/test_size.py
```

## Files

| file | purpose |
|------|---------|
| `mobilenetv2.py` | from-scratch MobileNet-v2 adapted for 32x32 CIFAR-10 |
| `pretrained_model.py` | torchvision pretrained MobileNet-v2 adapted for CIFAR-10 |
| `train_pretrained.py` / `compress_pretrained.py` | pretrained-variant pipeline |
| `data.py` | CIFAR-10 loaders + calibration subset |
| `train.py` | FP32 baseline training |
| `quant.py` | quantizers, BN folding, quant modules, calibration |
| `qat.py` | QAT fine-tuning (with optional distillation) |
| `sensitivity.py` | per-layer sensitivity + mixed-precision allocation |
| `pruning.py` / `prune.py` | channel pruning + fine-tune |
| `huffman.py` | from-scratch Huffman coder |
| `pack.py` | write/read the real compressed `.bin` |
| `size.py` | size accounting + compression ratios |
| `compress.py` | main compression entry point |
| `sweep.py` | wandb sweep for the parallel-coordinates chart |
