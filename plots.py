# make the loss / accuracy png curves from the history json that train.py saves
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_history(history, outdir='curves'):
    os.makedirs(outdir, exist_ok=True)
    epochs = [h['epoch'] for h in history]

    plt.figure()
    plt.plot(epochs, [h['train_loss'] for h in history])
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.title('Training loss')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(epochs, [h['train_acc'] for h in history], label='train')
    plt.plot(epochs, [h['test_acc'] for h in history], label='test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return os.path.join(outdir, 'loss.png'), os.path.join(outdir, 'accuracy.png')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--history', default='curves/history.json')
    p.add_argument('--outdir', default='curves')
    args = p.parse_args()
    with open(args.history) as f:
        hist = json.load(f)
    loss_png, acc_png = plot_history(hist, args.outdir)
    print(f"wrote {loss_png} and {acc_png}")
