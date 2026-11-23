"""复现作者代码关联分数，并计算滞后相关。"""
import numpy as np


def author_code_score(source, target, *, lag=1, bins=10):
    """Replicate the author's difference of joint entropies (not standard TE)."""
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) <= lag:
        raise ValueError("收益率长度不匹配或不足。")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("收益率包含缺失或非有限值。")
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1 or bins != 10:
        raise ValueError("当前仅支持正整数滞后与作者的 10 个分箱。")
    xb = np.digitize(x, bins=np.linspace(x.min(), x.max(), bins)) - 1
    yb = np.digitize(y, bins=np.linspace(y.min(), y.max(), bins)) - 1
    xp, yp, yf = xb[:-lag], yb[:-lag], yb[lag:]
    p2 = np.histogram2d(yf, yp, bins=bins)[0] + 1e-10
    p3 = np.histogramdd([yf, xp, yp], bins=bins)[0] + 1e-10
    p2 /= p2.sum()
    p3 /= p3.sum()
    return float(-np.sum(p2 * np.log2(p2)) + np.sum(p3 * np.log2(p3)))


def lagged_correlation(source, target, *, lag=1):
    x, y = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) <= lag + 1:
        raise ValueError("收益率长度不匹配或不足。")
    a, b = x[:-lag], y[lag:]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("收益率包含缺失或非有限值。")
    if np.std(a) == 0 or np.std(b) == 0:
        raise ValueError("常数序列无法计算相关系数。")
    return float(np.corrcoef(a, b)[0, 1])
