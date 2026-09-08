# -*- coding: utf-8 -*-
"""Builds the CS6886 Assignment 2 answer PDF (same layout as my Assignment 1).

Run:  python report/make_report.py
Screenshots go in report/snips/ (see the list printed at the end / the README).
Any screenshot that is missing is shown as a labelled placeholder box so the PDF
still builds; drop the real image in with the given filename and re-run.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, PageBreak, Image,
                                KeepTogether)
from reportlab.lib.enums import TA_CENTER
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from PIL import Image as PILImage

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
SNIP = os.path.join(BASE, 'snips')
OUT  = os.path.join(BASE, 'CS6886_Assignment2_Answers.pdf')
LOGO = os.path.join(BASE, 'iitm-logo.png')
STUDENT = 'Vipin Kumar'
ROLL    = 'NS26Z171'

styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=15, spaceAfter=6,
                    textColor=colors.HexColor('#12355b'))
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12.5, spaceBefore=8,
                    spaceAfter=4, textColor=colors.HexColor('#1f4e79'))
BODY = ParagraphStyle('BODY', parent=styles['BodyText'], fontSize=9.6, leading=13,
                      spaceAfter=5, alignment=4)
SMALL = ParagraphStyle('SMALL', parent=styles['BodyText'], fontSize=8.3, leading=10.5,
                       textColor=colors.HexColor('#444444'))
NOTE = ParagraphStyle('NOTE', parent=BODY, fontSize=8.8, leading=11.5,
                      leftIndent=6, textColor=colors.HexColor('#333333'))
CELL = ParagraphStyle('CELL', parent=styles['BodyText'], fontSize=8.2, leading=10)
CELLB = ParagraphStyle('CELLB', parent=CELL, fontName='Helvetica-Bold')
CAP  = ParagraphStyle('CAP', parent=styles['BodyText'], fontSize=8, leading=10,
                      alignment=TA_CENTER, textColor=colors.HexColor('#555555'),
                      spaceBefore=2, spaceAfter=8)

story = []
def P(t, s=BODY): story.append(Paragraph(t, s))
def SP(h=4): story.append(Spacer(1, h))

_figno = [0]
def _img_or_placeholder(path, w, h):
    if os.path.exists(path):
        iw, ih = PILImage.open(path).size
        sc = min(w/iw, h/ih)
        im = Image(path, width=iw*sc, height=ih*sc); im.hAlign = 'CENTER'
        return im
    ph = Table([[Paragraph('screenshot placeholder<br/><b>%s</b>' %
                 os.path.basename(path), CAP)]], colWidths=[w], rowHeights=[h*0.5])
    ph.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#eef1f5')),
                            ('BOX',(0,0),(-1,-1),0.6,colors.HexColor('#b8c4d0')),
                            ('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    return ph

def fig(relpath, caption, maxw=150*mm, maxh=92*mm, base=None):
    path = relpath if os.path.isabs(relpath) else os.path.join(base or SNIP, relpath)
    im = _img_or_placeholder(path, maxw, maxh)
    _figno[0] += 1
    cap = Paragraph('<b>Figure %d.</b> %s' % (_figno[0], caption), CAP)
    story.append(KeepTogether([im, cap]))

def figrow(items, maxw=84*mm, maxh=66*mm, base=None):
    cells = []
    for relpath, cap in items:
        path = relpath if os.path.isabs(relpath) else os.path.join(base or SNIP, relpath)
        im = _img_or_placeholder(path, maxw, maxh)
        _figno[0] += 1
        c = Paragraph('<b>Figure %d.</b> %s' % (_figno[0], cap), CAP)
        cells.append([im, c])
    data = [[cells[0][0], cells[1][0]], [cells[0][1], cells[1][1]]]
    t = Table(data, colWidths=[88*mm, 88*mm])
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                           ('ALIGN',(0,0),(-1,-1),'CENTER'),
                           ('TOPPADDING',(0,0),(-1,-1),2),
                           ('BOTTOMPADDING',(0,0),(-1,-1),2)]))
    story.append(t)

def mktable(data, colw, header=True, fs=8.2, align=None):
    t = Table(data, colWidths=colw, repeatRows=1 if header else 0)
    cmds = [('FONTSIZE',(0,0),(-1,-1),fs),
            ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#b8c4d0')),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('TOPPADDING',(0,0),(-1,-1),2.5),('BOTTOMPADDING',(0,0),(-1,-1),2.5),
            ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]
    if header:
        cmds += [('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d6e0ee')),
                 ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                 ('LINEBELOW',(0,0),(-1,0),0.6,colors.HexColor('#7f97b5')),
                 ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f9')])]
    if align:
        for col,a in align.items(): cmds.append(('ALIGN',(col,0),(col,-1),a))
    t.setStyle(TableStyle(cmds))
    return t

DARK = colors.HexColor('#1a1a1a')
ARROW = colors.HexColor('#555555')

def _box(d, x, y, w, h, lines, fs=6.7):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=colors.HexColor('#eef1f5'),
               strokeColor=colors.HexColor('#7f97b5'), strokeWidth=0.9))
    m = len(lines)
    top = y + h / 2 + (m - 1) * 4.5 - 2.5
    for j, ln in enumerate(lines):
        d.add(String(x + w / 2, top - j * 9, ln, textAnchor='middle', fontSize=fs,
                     fontName='Helvetica-Bold' if j == 0 else 'Helvetica', fillColor=DARK))

def _arrow(d, x, y, gap):
    d.add(Line(x, y, x + gap - 3, y, strokeColor=ARROW, strokeWidth=1))
    d.add(Polygon([x + gap - 3, y - 3, x + gap, y, x + gap - 3, y + 3],
                  fillColor=ARROW, strokeColor=ARROW))

def flow_diagram(steps, width=500, box_h=52, gap=16):
    n = len(steps)
    bw = (width - gap * (n - 1)) / float(n)
    d = Drawing(width, box_h + 6); d.hAlign = 'CENTER'
    for i, lines in enumerate(steps):
        x = i * (bw + gap)
        _box(d, x, 3, bw, box_h, lines)
        if i < n - 1:
            _arrow(d, x + bw, 3 + box_h / 2, gap)
    return d

def quant_diagram(width=500, box_h=40, gap=14, row_gap=24, labelw=54):
    d = Drawing(width, box_h * 2 + row_gap + 6); d.hAlign = 'CENTER'
    def row(y, label, steps):
        d.add(String(0, y + box_h / 2 - 3, label, fontSize=7.5,
                     fontName='Helvetica-Bold', fillColor=colors.HexColor('#1f4e79')))
        n = len(steps); bw = (width - labelw - gap * (n - 1)) / float(n)
        for i, lines in enumerate(steps):
            x = labelw + i * (bw + gap)
            _box(d, x, y, bw, box_h, lines, fs=6.2)
            if i < n - 1:
                _arrow(d, x + bw, y + box_h / 2, gap)
    row(box_h + row_gap, 'Weights',
        [['W (fp32)'], ['scale s = clip99.9(|W|)', '/ (2^(b-1)-1)', 'per output channel'],
         ['q = round(W / s)', 'clamp +/-(2^(b-1)-1)'], ['W = q * s', '(STE gradient in QAT)']])
    row(0, 'Activations',
        [['X after ReLU6'], ['calibrate min/max', '-> scale s, zero-point z'],
         ['q = round(X/s + z)', 'clamp 0 .. 2^b-1'], ['X = (q - z) * s']])
    return d

def diagram(dr, caption):
    _figno[0] += 1
    cap = Paragraph('<b>Figure %d.</b> %s' % (_figno[0], caption), CAP)
    story.append(KeepTogether([dr, cap]))

# ================= TITLE PAGE =================
TITLE_BIG = ParagraphStyle('TB', parent=styles['Title'], fontSize=20, leading=24,
                           alignment=TA_CENTER, textColor=colors.HexColor('#7a1f1f'))
INST = ParagraphStyle('INST', parent=styles['Title'], fontSize=15, leading=19,
                      alignment=TA_CENTER, textColor=colors.HexColor('#12355b'))
SUB = ParagraphStyle('SUB', parent=styles['Normal'], fontSize=12.5, leading=17, alignment=TA_CENTER)
SUB2 = ParagraphStyle('SUB2', parent=styles['Normal'], fontSize=11, leading=16,
                      alignment=TA_CENTER, textColor=colors.HexColor('#333333'))

if os.path.exists(LOGO):
    _lw,_lh = PILImage.open(LOGO).size
    _logo = Image(LOGO, width=42*mm, height=42*mm*_lh/_lw); _logo.hAlign='CENTER'
    story.append(Spacer(1,12*mm)); story.append(_logo)
story.append(Spacer(1,6*mm))
P('INDIAN INSTITUTE OF TECHNOLOGY MADRAS', INST)
P('Department of Computer Science and Engineering', SUB2)
story.append(Spacer(1,3*mm))
story.append(HRFlowable(width='72%', thickness=1.1, color=colors.HexColor('#7a1f1f')))
story.append(Spacer(1,6*mm))
P('CS6886 : Systems Engineering for Deep Learning', SUB)
story.append(Spacer(1,4*mm))
P('Assignment 2', TITLE_BIG)
story.append(Spacer(1,3*mm))
story.append(HRFlowable(width='72%', thickness=1.1, color=colors.HexColor('#7a1f1f')))
story.append(Spacer(1,12*mm))
P('Submitted by', SUB2)
P('<b>%s</b>' % STUDENT, SUB)
P('Roll No: <b>%s</b>' % ROLL, SUB)
story.append(Spacer(1,10*mm))
P('September 2026', SUB2)
story.append(PageBreak())

# ---------------- Intro + required-numbers summary ----------------
P('Assignment 2: Solutions', H1)
story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#1f4e79')))
SP(4)
P('I trained MobileNet-v2 on CIFAR-10 from scratch and then compressed it using the three '
  'techniques we covered in class - <b>pruning, quantization and knowledge distillation</b>. '
  'The quantization and the entropy coding are written by hand (no library quantizers), as the '
  'assignment asks. My GitHub repository with exact commands, environment and seeds is linked in '
  'Question&nbsp;5. The headline numbers the assignment asks for are collected below and derived in '
  'the later sections.', BODY)
SP(3)
summ = [['Item','Value'],
 ['Accuracy of the model without any compression','94.88% (top-1)'],
 ['Best compression ratio of the model','18.99&times;  (at 93.67% accuracy)'],
 ['Best compression ratio of the weights','19.38&times;'],
 ['Best compression ratio of the activations','7.39&times; (6-bit quant 5.33&times; &times; pruning 1.39&times;)'],
 ['Storage overheads (scales + int8 bias + Huffman tables)','~38 KB'],
 ['Final approximated model size after compression','0.45 MB  (from 8.53 MB)']]
story.append(mktable([[Paragraph(c, CELLB if (i>0 and j==1) else CELL) for j,c in enumerate(r)]
                      for i,r in enumerate(summ)], [104*mm, 66*mm]))
SP(2)
P('The single configuration behind these numbers is <b>30% channel pruning + mixed-precision '
  'quantization-aware training (average 3-bit weights, 6-bit activations) + Huffman coding</b>, with '
  'distillation from the FP32 model. All ratios are against the uncompressed FP32 model (8.53 MB). '
  'The Wandb parallel-coordinates chart for the full sweep is in Question&nbsp;3 (Figure&nbsp;5).', NOTE)
SP(4)

# ================= Q1 =================
P('Question 1: Training Baseline (20 pts)', H2)
P('<b>(a) Data preparation and transforms.</b> I use the standard CIFAR-10 pipeline. Images are '
  'normalized per channel with mean = (0.4914, 0.4822, 0.4465) and std = (0.2470, 0.2435, 0.2616). '
  'For the training set I augment with a random 32&times;32 crop (4-pixel reflection padding), a '
  'random horizontal flip, the AutoAugment CIFAR-10 policy, and finally a Cutout-style random erase '
  '(p = 0.25). The test set only gets ToTensor + normalize. The augmentation is fairly aggressive on '
  'purpose - MobileNet-v2 has a lot of capacity for CIFAR, so without it the model overfits.', BODY)
P('<b>(b) Model configuration and training strategy.</b> The network is MobileNet-v2 [6] with width '
  'multiplier 1.0, the usual inverted-residual blocks (expansion factor 6), ReLU6 activations, '
  'BatchNorm after every convolution, and dropout 0.2 before the classifier. I wrote it from scratch '
  'and adapted it for 32&times;32 inputs by setting the stem convolution stride and the first '
  'inverted-residual stage stride to 1 (the ImageNet version downsamples 32&times;, which is far too '
  'much for CIFAR). Training: SGD with momentum 0.9 and Nesterov, weight decay 5e-4, initial learning '
  'rate 0.05 with cosine annealing, label smoothing 0.1, batch size 128, 150 epochs, seed 42.', BODY)
P('<b>(c) Result and curves.</b> The final test top-1 accuracy is <b>94.88%</b> (best over the run; '
  'the last epoch is 94.85%). The loss and accuracy curves are below. Train and test accuracy stay '
  'close together the whole time (train ends at ~94.4%), so there is very little overfitting, which '
  'tells me the augmentation and weight decay are doing their job.', BODY)
figrow([(os.path.join(PROJ,'curves','loss.png'), 'Training loss vs epoch (from-scratch baseline).'),
        (os.path.join(PROJ,'curves','accuracy.png'), 'Train / test accuracy vs epoch; best test = 94.88%.')])
P('<b>Failure modes I observed.</b> The main one shows up later during compression: the '
  '<b>depthwise convolution layers are by far the most sensitive</b> to quantization, and under '
  'plain post-training quantization there is a sharp accuracy cliff below 3-bit weights '
  '(94.9% &rarr; 76% at 3-bit, &rarr; 13% at 2-bit). Activations, on the other hand, are bounded by '
  'ReLU6 so they quantize cleanly down to 6 bits. During training itself the only instability was '
  'that too high a learning rate early on stalled the run, which the cosine schedule + warmup-like '
  'low start fixed.', BODY)
SP(4)

# ================= Q2 =================
P('Question 2: Model Compression Implementation (30 pts)', H2)
P('The overall pipeline follows the "prune then quantize then entropy-code" recipe from class:', BODY)
diagram(flow_diagram([
    ['FP32 model', '94.88% top-1', '8.53 MB'],
    ['Channel prune', 'BN-gamma', '30% per block'],
    ['Mixed QAT', '3-bit W / 6-bit A', 'per-channel + clip', '+ distillation'],
    ['Huffman coding', 'of weight indices'],
    ['Compressed', '0.45 MB', '93.67% @ 18.99x']]),
    'The compression pipeline. Pruning and quantization stack; Huffman then entropy-codes the '
    'quantized indices.')
P('<b>(a) Configurable compression method and design choices.</b> My quantizer is uniform '
  '(linear) affine quantization, exactly the scale-and-zero-point form from the linear-quantization '
  'lecture: an integer code q = round(x/s) + z, and the real value is recovered as s&middot;(q &minus; z). '
  'For <b>weights</b> I use <b>symmetric</b> quantization (zero point = 0) on a <b>per-output-channel</b> '
  'basis - the lecture showed per-channel has lower quantization error than per-tensor, and this '
  'matters a lot here because the depthwise channels have very different weight ranges, so one shared '
  'per-tensor scale destroys them. For <b>activations</b> I use <b>asymmetric</b>, <b>per-tensor</b>, '
  'unsigned quantization because they sit after ReLU6 (non-negative, bounded). Since activation ranges '
  'are not known ahead of time, I calibrate the min/max on a small set of training images before '
  'freezing the scale, again as described in the weights-vs-activations slide. Both weight and '
  'activation bit-widths are command-line options, and a mixed-precision mode picks a different '
  'bit-width per layer (details in part b). Two paths are supported: post-training quantization (PTQ) '
  'and quantization-aware training (QAT); QAT uses the straight-through estimator so the discrete '
  'round has a usable gradient, and the lecture result that QAT beats PTQ at low bits is exactly what '
  'I see.', BODY)
diagram(quant_diagram(),
    'The two quantization paths. Weights: symmetric, per-output-channel (scale s, no zero-point). '
    'Activations: asymmetric, per-tensor after ReLU6 (scale s and zero-point z from calibration). '
    'b is the bit-width; QAT trains through the round with the straight-through estimator (STE).')
P('On top of quantization I add the other two techniques from the "prune &rarr; quantize &rarr; '
  'entropy code" methodology (the Deep Compression pipeline [1] from the K-means-quantization '
  'lecture):', BODY)
for b in [
 '<b>Channel pruning</b> (structured, from the pruning-granularity lecture): I prune whole expansion '
 'channels, which is the hardware-friendly granularity, using the network-slimming criterion [3] - '
 'the magnitude of each channel&rsquo;s BatchNorm scaling factor (gamma) as the importance score.',
 '<b>Huffman coding</b> of the quantized weight indices (the entropy-coding step of Deep Compression), '
 'written from scratch, to squeeze the indices below the fixed bit-width.',
 '<b>Knowledge distillation</b> during QAT: the FP32 model is the teacher and the quantized/pruned '
 'model is the student, trained with a KL-divergence term on the softened logits (temperature 4), '
 'as in the KD lecture [5]. This is what lets me push to 3-bit and still stay near baseline.']:
    P('&bull;&nbsp; ' + b, NOTE)
P('<b>(b) How it is applied to MobileNet-v2 (which layers, exceptions).</b> BatchNorm is first '
  'folded into the preceding convolution, so I quantize the weights that are actually used at '
  'inference. Every convolution and the final linear layer have their weights quantized, and every '
  'ReLU6 output is quantized. Under mixed precision the sensitive layers - the first '
  'convolution, the classifier, and the depthwise layers - are kept at higher bit-widths, while '
  'the robust pointwise layers go lower; the per-layer budget is chosen by a small sensitivity test '
  '(drop one layer to 2-bit, measure the accuracy loss, rank them), which is the layer-sensitivity '
  'idea from the channel-pruning slide applied to bit-width - a simpler version of the '
  'sensitivity-based mixed precision in HAWQ-v2 [7], Pack-PTQ [8] and dMX [9]. Pruning is applied '
  'only to the '
  '<b>internal expansion channels</b> of each inverted-residual block (the hidden dimension between '
  'the expand and project 1&times;1 convolutions). These are local to the block, so removing them '
  'does not change the block&rsquo;s input/output width and leaves the residual add untouched - '
  'the first block (no expansion) is left alone. I prune 30% per block, which takes the model from '
  '2.24M to 1.70M parameters, and after the distillation fine-tune the pruned FP32 model actually '
  'reaches 95.28% (slightly above the dense baseline, because the fine-tune regularizes the smaller '
  'network).', BODY)
P('<b>(c) Storage overheads.</b> Using the model-size = (#parameters &times; bit-width) view from '
  'class, the quantized indices are the bulk of the size, but the metadata has to be counted too. My '
  'size accounting includes: the per-channel weight <b>scales</b> (fp16, one per output channel), the '
  'folded <b>biases</b> (stored in int8 with one fp16 scale per layer, which halves them versus '
  'fp16), and the <b>Huffman code tables</b> (one (symbol, code-length) entry per distinct value per '
  'layer). For the reported best configuration these come to about <b>38&nbsp;KB</b> in total '
  '(scales 25.7&nbsp;KB + biases 12.9&nbsp;KB + tables 0.4&nbsp;KB), and this is already included in '
  'the 0.45&nbsp;MB final size - i.e. the compression ratios are not "cheating" by ignoring '
  'metadata. To check the size is real and not just a formula, I also serialise the compressed '
  'model to an actual binary (Huffman bitstream + fp16 scales + int8 biases + activation '
  'scale/zero-point): the file is <b>0.45&nbsp;MB on disk</b>, and reloading and decoding it re-runs '
  'the test set at <b>93.66%</b> (the ~0.1% gap versus 93.67% is just from storing the scales in '
  'fp16).', BODY)
ov = [['Component','Stored as','Bytes'],
      ['Quantized weight indices (avg 3-bit, Huffman)','variable-length codes','~432,000'],
      ['Per-channel weight scales','fp16 x #out-channels','25,652'],
      ['Folded biases','int8 + per-layer fp16 scale','12,932'],
      ['Huffman code tables','(symbol, length) pairs','371'],
      ['Total compressed model','','~0.45 MB']]
story.append(mktable([[Paragraph(c, CELLB if (i==len(ov)-1) else CELL) for c in r] for i,r in enumerate(ov)],
                     [78*mm,44*mm,30*mm], align={2:'RIGHT'}))
SP(4)

# ================= Q3 =================
P('Question 3: Compression Results', H2)
P('I swept the compression at different levels - weight bit-width, activation bit-width, PTQ '
  'vs QAT, and uniform vs mixed precision - and logged everything to Wandb. The table lists the '
  'main operating points and the parallel-coordinates chart (Figure&nbsp;5) shows the whole sweep at '
  'once, with each run as a line across the bit-width, method, compression-ratio and accuracy axes. '
  'The uncompressed baseline is 94.88% at 8.53&nbsp;MB.', BODY)
res = [['Method','Weights','Act.','Accuracy','Model CR','Weight CR','Act. CR','Size'],
 ['PTQ','8-bit','8','94.79%','4.41x','4.45x','4.00x','1.92 MB'],
 ['PTQ','6-bit','8','94.58%','6.13x','6.23x','4.00x','1.38 MB'],
 ['PTQ','4-bit','8','93.23%','10.03x','10.35x','4.00x','0.84 MB'],
 ['PTQ','3-bit','8','76.18%','14.76x','15.52x','4.00x','0.57 MB'],
 ['PTQ','2-bit','8','13.43%','23.74x','25.92x','4.00x','0.36 MB'],
 ['QAT','4-bit','6','94.41%','10.03x','10.35x','5.33x','0.84 MB'],
 ['QAT','3-bit','6','92.87%','14.76x','15.52x','5.33x','0.57 MB'],
 ['QAT','mixed ~2.8','6','92.04%','16.14x','17.08x','5.33x','0.52 MB'],
 ['Prune+QAT','mixed 3.0','6','93.67%','18.99x','19.38x','7.39x','0.45 MB']]
story.append(mktable([[Paragraph(c, CELLB if (i==len(res)-1) else CELL) for c in r] for i,r in enumerate(res)],
                     [20*mm,18*mm,10*mm,20*mm,18*mm,18*mm,16*mm,18*mm],
                     align={j:'CENTER' for j in range(2,8)}))
SP(3)
fig('wandb_parallel_coordinates.png',
    'Wandb parallel-coordinates plot of the compression sweep (project cs6886-a2). Each line is one '
    'run; the axes are average weight bits, activation bits, model compression ratio, method '
    '(PTQ/QAT), model size (MB) and top-1 accuracy (colour). The dark lines that plunge to low '
    'accuracy are the 2-bit runs.')
P('<b>What the sweep shows.</b> PTQ is basically lossless down to 6-bit weights and only loses '
  '~1.7% at 4-bit, but it falls off a cliff at 3-bit (76%) and collapses at 2-bit (13%). QAT '
  'recovers the 3-bit case all the way back to ~93%, which is the "QAT &gt; PTQ" result from the '
  'lecture. Uniform 2-bit is unusable even with QAT because the depthwise layers cannot be '
  'represented in 2 bits - this is exactly why mixed precision helps: it spends the bit budget '
  'on the sensitive layers and reaches a lower <i>average</i> bit-width (~2.8) while keeping ~92%. '
  'Finally, pruning stacks on top: 30% channel pruning followed by mixed 3-bit QAT gives the best '
  'point overall, 18.99&times; at 93.67%, which beats every quantization-only setting on both '
  'compression and accuracy. This matches the slide that said "pruning followed by quantization '
  'provides the best model compression".', BODY)
SP(4)

# ================= Q4 =================
P('Question 4: Compression Analysis', H2)
P('The one configuration I would submit is <b>30% channel pruning + mixed-precision QAT '
  '(average 3-bit weights, 6-bit activations) + per-channel weight quantization + int8 biases + '
  'Huffman coding, with distillation</b>. It gives the highest compression while staying within about '
  '1.2% of the baseline. All ratios are against the uncompressed FP32 model (8.53&nbsp;MB).', BODY)
for b in [
 '<b>(a) Weight compression ratio: 19.38&times;.</b> FP32 weights (8.47&nbsp;MB) versus the pruned, '
 'Huffman-coded indices plus per-channel scales and code tables.',
 '<b>(b) Activation compression ratio: 7.39&times;.</b> I measured activations per inference at '
 'batch size 1 by summing the element count of every quantized activation tensor across the network '
 '(a forward hook on each activation quantizer). FP32 counts 32 bits per element; the compressed '
 'model counts 6 bits per element plus one scale + zero-point per tensor, over the pruned (smaller) '
 'feature maps. So the 7.39&times; is pruning shrinking the feature maps combined with 6-bit '
 'activation quantization (quantization alone gives 32/6 = 5.33&times;).',
 '<b>(c) Accuracy of the quantized model: 93.67%</b> (1.21% below the 94.88% FP32 baseline).',
 '<b>(d) Final model size: 0.45&nbsp;MB</b>, down from 8.53&nbsp;MB - an 18.99&times; overall '
 'model compression ratio.']:
    P('&bull;&nbsp; ' + b, NOTE)
P('If accuracy mattered more than size, the un-pruned QAT 4-bit / 6-bit point is a safer choice: '
  '94.41% (only 0.47% below baseline) at 10.03&times;. I picked the pruned mixed-3-bit point as the '
  'headline because the assignment rewards the compression ratio and 93.67% is still close to the '
  'baseline.', BODY)
SP(2)
fig('terminal_best_config.png',
    'Terminal output for the best config (pruned + mixed 3-bit QAT). Note the script prints the ratio '
    'against the pruned FP32 model (6.42&nbsp;MB &rarr; 0.45&nbsp;MB = 14.28&times;); measured against '
    'the original 8.53&nbsp;MB dense model, the same 0.45&nbsp;MB gives the 18.99&times; reported '
    'above. The activation ratio printed (5.33&times;) is the 6-bit quantization part; with the '
    'pruned feature maps it is 7.39&times; overall.', maxw=140*mm, maxh=88*mm)
SP(2)

# ================= Q5 =================
P('Question 5: Reproducibility &amp; Repository (10 pts)', H2)
P('<b>(a)</b> The codebase is modular with training, evaluation and compression kept separate: '
  '<font face="Courier">train.py</font> (baseline), <font face="Courier">quant.py</font> '
  '(quantizers, BN folding, calibration), '
  '<font face="Courier">qat.py</font>, <font face="Courier">sensitivity.py</font> (mixed precision), '
  '<font face="Courier">pruning.py</font> / <font face="Courier">prune.py</font>, '
  '<font face="Courier">huffman.py</font>, <font face="Courier">size.py</font> (size accounting), '
  '<font face="Courier">pack.py</font> (writes/reads the real compressed .bin) and '
  '<font face="Courier">compress.py</font> (driver). Everything is commented.', BODY)
P('<b>(b)</b> The README lists the exact commands, the environment (Python 3.11, torch 2.11 + CUDA '
  '12.8, torchvision 0.26), pinned dependency versions in <font face="Courier">requirements.txt</font>, '
  'and the fixed seed (42) used everywhere. A small test suite checks the Huffman round-trip, the '
  'size accounting, and that per-channel beats per-tensor quantization.', BODY)
P('<b>(c) GitHub repository:</b> '
  '<font face="Courier">https://github.com/kumaram-vipinam200901/CS6886_A2_quantization</font>', BODY)
SP(4)

# ================= References =================
P('References', H2)
refs = [
 'S. Han, H. Mao and W. J. Dally. "Deep Compression: Compressing Deep Neural Networks with Pruning, '
 'Trained Quantization and Huffman Coding." ICLR, 2016.',
 'S. Han, J. Pool, J. Tran and W. Dally. "Learning both Weights and Connections for Efficient Neural '
 'Networks." NeurIPS, 2015.',
 'Z. Liu, J. Li, Z. Shen, G. Huang, S. Yan and C. Zhang. "Learning Efficient Convolutional Networks '
 'through Network Slimming." ICCV, 2017.',
 'B. Jacob et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only '
 'Inference." CVPR, 2018.',
 'G. Hinton, O. Vinyals and J. Dean. "Distilling the Knowledge in a Neural Network." NeurIPS '
 'Workshop, 2014.',
 'M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen. "MobileNetV2: Inverted Residuals and '
 'Linear Bottlenecks." CVPR, 2018.',
 'Z. Dong, Z. Yao, D. Arfeen, A. Gholami, M. W. Mahoney and K. Keutzer. "HAWQ-V2: Hessian Aware '
 'trace-Weighted Quantization of Neural Networks." NeurIPS, 2020.',
 'C. Li, R. Jiang, Z. Song, P. Yu, Y. Zhang and Y. Guo. "Pack-PTQ: Advancing Post-training '
 'Quantization of Neural Networks by Pack-wise Reconstruction." arXiv:2505.00259, 2025.',
 'G. Franco, I. Colbert, P. Monteagudo-Lago, F. Marty and N. Fraser. "dMX: Differentiable '
 'Mixed-Precision Assignment for Low-Precision Floating-Point Formats." AMD, 2025.',
 'Course lectures, CS6886 (Prof. Gopalakrishnan Srinivasan): pruning granularity and criterion, '
 'linear quantization, K-means quantization, and knowledge distillation.']
for i, r in enumerate(refs):
    P('[%d]&nbsp; %s' % (i+1, r), NOTE)

# ---------------- Footer ----------------
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(colors.HexColor('#888888'))
    canvas.drawString(16*mm, 8*mm, '%s (%s)  |  CS6886 Assignment 2' % (STUDENT, ROLL))
    canvas.drawRightString(A4[0]-16*mm, 8*mm, 'Page %d' % doc.page)
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm,
                        topMargin=14*mm, bottomMargin=16*mm,
                        title='CS6886 Assignment 2 - %s (%s)' % (STUDENT, ROLL),
                        author=STUDENT)
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print('WROTE', OUT)
