from __future__ import annotations
import numpy as np

def zscore(x, axis=None, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (std + eps)

def group_zscore(values, groups):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    out = np.zeros_like(values, dtype=np.float64)
    for gid in np.unique(groups):
        idx = (groups == gid)
        out[idx] = zscore(values[idx])
    return out

def delta_mos(mos, is_ref):
    mos = np.asarray(mos, dtype=np.float64)
    is_ref = np.asarray(is_ref).astype(bool)
    if not is_ref.any():
        raise ValueError("需要在每组内包含参考样本以计算 ΔMOS")
    ref_val = mos[is_ref].mean()
    return ref_val - mos

def logistic_5p(x, b1, b2, b3, b4, b5):
    x = np.asarray(x, dtype=np.float64)
    return b1 * (0.5 - 1.0 / (1 + np.exp(b2 * (x - b3)))) + b4 * x + b5

def logistic_4p(x, b1, b2, b3, b4):
    x = np.asarray(x, dtype=np.float64)
    return b1 * (0.5 - 1.0 / (1 + np.exp(b2 * (x - b3)))) + b4

def fit_logistic(y_pred, y_true, use_5p=True):
    try:
        from scipy.optimize import curve_fit
        func = logistic_5p if use_5p else logistic_4p
        p0 = [1, 1, np.median(y_pred), 0, 0] if use_5p else [1, 1, np.median(y_pred), 0]
        beta, _ = curve_fit(func, np.asarray(y_pred), np.asarray(y_true), p0=p0, maxfev=20000)
        mapped = func(y_pred, *beta)
        return mapped, beta
    except Exception:
        return np.asarray(y_pred, dtype=np.float64), None

def plcc(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())

def srcc(a, b):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    except Exception:
        from scipy.stats import rankdata
        ra = rankdata(a); rb = rankdata(b)
        return plcc(ra, rb)

def krcc(a, b):
    try:
        from scipy.stats import kendalltau
        return float(kendalltau(a, b).correlation)
    except Exception:
        return srcc(a, b)
