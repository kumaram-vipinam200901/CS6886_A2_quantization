# Quantization-aware training: fine-tune the quantized model (STE) so the
# weights adapt to the quantization noise. Activation ranges stay frozen.
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import evaluate


def train_qat(model, train_loader, test_loader, device, epochs=5, lr=5e-4,
              log_fn=None, teacher=None, kd_alpha=0.5, kd_temp=4.0):
    """Fine-tune a quantized model. If `teacher` (an FP32 model) is given, we add
    a knowledge-distillation loss so the low-bit student mimics the teacher's
    soft predictions, which recovers extra accuracy at aggressive bit-widths."""
    model.to(device)
    if teacher is not None:
        teacher.to(device).eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
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
                kd = F.kl_div(F.log_softmax(out / kd_temp, dim=1),
                              F.softmax(t_out / kd_temp, dim=1),
                              reduction='batchmean') * (kd_temp ** 2)
                loss = kd_alpha * kd + (1 - kd_alpha) * loss
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        print(f"[QAT] epoch {epoch + 1}/{epochs}  loss={running:.1f}  test_acc={acc:.2f}%")
        if log_fn:
            log_fn({'qat_epoch': epoch + 1, 'qat_loss': running, 'qat_test_acc': acc})
    return model, best_acc
