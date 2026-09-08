# Structured channel pruning: prune the internal expansion channels of each
# inverted-residual block (local to the block, so residual connections are safe).
# Importance = BatchNorm scale (gamma) magnitude of the expand layer (network
# slimming). Keep the top channels, rebuild a smaller model, copy weights over.
import torch

from mobilenetv2 import mobilenet_v2, InvertedResidual


def _blocks(model):
    return [m for m in model.features if isinstance(m, InvertedResidual)]


def _has_expand(block):
    # expand present -> conv = [expand, depthwise, project_conv, project_bn] (len 4)
    # expand absent  -> conv = [depthwise, project_conv, project_bn]          (len 3)
    return len(block.conv) == 4


def _round8(x):
    return max(8, int(round(x / 8)) * 8)


def pruned_hidden_dims(dense, ratio):
    """Return the list of hidden widths after pruning `ratio` of each block."""
    dims = []
    for b in _blocks(dense):
        if _has_expand(b):
            keep = _round8(b.hidden_dim * (1.0 - ratio))
            dims.append(min(b.hidden_dim, keep))
        else:
            dims.append(b.hidden_dim)  # first block: not pruned
    return dims


@torch.no_grad()
def make_pruned_model(dense, ratio, num_classes=10, dropout=0.2):
    """Build a smaller model and copy the surviving weights from `dense`."""
    dims = pruned_hidden_dims(dense, ratio)
    pruned = mobilenet_v2(num_classes=num_classes, dropout=dropout, hidden_dims=dims)

    # first conv+bn and the final conv+bn are unchanged
    pruned.features[0].load_state_dict(dense.features[0].state_dict())
    pruned.features[-1].load_state_dict(dense.features[-1].state_dict())
    pruned.classifier.load_state_dict(dense.classifier.state_dict())

    d_blocks = _blocks(dense)
    p_blocks = _blocks(pruned)
    for db, pb in zip(d_blocks, p_blocks):
        if not _has_expand(db) or pb.hidden_dim == db.hidden_dim:
            pb.load_state_dict(db.state_dict())
            continue

        gamma = db.conv[0][1].weight.abs()          # expand BN scale
        keep = torch.sort(torch.topk(gamma, pb.hidden_dim).indices).values

        # expand conv + bn (output channels pruned)
        pb.conv[0][0].weight.copy_(db.conv[0][0].weight[keep])
        _copy_bn(pb.conv[0][1], db.conv[0][1], keep)
        # depthwise conv + bn (channels pruned)
        pb.conv[1][0].weight.copy_(db.conv[1][0].weight[keep])
        _copy_bn(pb.conv[1][1], db.conv[1][1], keep)
        # project conv (input channels pruned), project bn unchanged
        pb.conv[2].weight.copy_(db.conv[2].weight[:, keep])
        pb.conv[3].load_state_dict(db.conv[3].state_dict())

    return pruned, dims


def _copy_bn(dst, src, idx):
    dst.weight.copy_(src.weight[idx])
    dst.bias.copy_(src.bias[idx])
    dst.running_mean.copy_(src.running_mean[idx])
    dst.running_var.copy_(src.running_var[idx])
