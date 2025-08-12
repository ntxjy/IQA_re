import numpy as np
import matplotlib.pyplot as plt
from .mos_utils import fit_logistic

def plot_training_curves(history, keys=("loss",), title=None, out=None):
    ep = history.get("epoch", list(range(1, len(history.get(keys[0], []))+1)))
    plt.figure()
    for k in keys:
        if k in history:
            plt.plot(ep, history[k], label=k)
    plt.xlabel("Epoch"); plt.ylabel("Value")
    if title: plt.title(title)
    plt.legend(); plt.tight_layout()
    if out: plt.savefig(out, dpi=200); plt.close()

def plot_scatter_with_logistic(pred, mos, use_5p=True, title=None, out=None):
    pred = np.asarray(pred); mos = np.asarray(mos)
    mapped, _ = fit_logistic(pred, mos, use_5p=use_5p)
    plt.figure()
    plt.scatter(pred, mos, s=10, alpha=0.6, label="Raw")
    idx = np.argsort(pred); plt.plot(pred[idx], mapped[idx], linewidth=2, label="Logistic fit")
    plt.xlabel("Prediction"); plt.ylabel("MOS")
    if title: plt.title(title)
    plt.legend(); plt.tight_layout()
    if out: plt.savefig(out, dpi=200); plt.close()
