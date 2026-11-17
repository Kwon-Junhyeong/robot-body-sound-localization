"""Conformal prediction regions on the matching-score map, with an adaptive risk level."""
import numpy as np


def normalize(score_map):
    lo, hi = score_map.min(-1, keepdims=True), score_map.max(-1, keepdims=True)
    return (score_map - lo) / (hi - lo + 1e-12)


def nonconformity(score_map, true_idx):
    """s = 1 - normalized map value at the true direction."""
    return 1.0 - normalize(score_map)[np.arange(len(true_idx)), true_idx]


def threshold(s_cal, alpha):
    """q: the (1 - alpha) quantile of the calibration scores, with the finite-sample correction."""
    n = len(s_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(s_cal)[min(max(k, 1), n) - 1]


def region(score_map, q):
    """Directions with s <= q. They hold the true direction with probability 1 - alpha over answers."""
    return 1.0 - normalize(score_map) <= q


def region_share(mask, grid):
    """Share of the sphere covered by the region (grid points weighted by the cosine of elevation)."""
    w = np.sqrt(np.clip(1.0 - grid[:, 2] ** 2, 0.0, 1.0))
    return (mask * w).sum(-1) / w.sum()


class AdaptiveCP:
    """alpha_t+1 = alpha_t + gamma (alpha - err_t), where err_t = 1 if the true direction fell outside the region."""

    def __init__(self, s_cal, alpha, gamma=0.02):
        self.s_cal, self.alpha, self.gamma, self.alpha_t = np.asarray(s_cal), alpha, gamma, alpha

    def q(self):
        return threshold(self.s_cal, float(np.clip(self.alpha_t, 1e-4, 1.0)))

    def update(self, err):
        self.alpha_t += self.gamma * (self.alpha - float(err))


def answer(mask, grid, max_share):
    """Answer when the region is narrow, abstain when it is wide."""
    return region_share(mask, grid) <= max_share
