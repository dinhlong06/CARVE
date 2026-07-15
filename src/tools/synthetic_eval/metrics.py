import numpy as np

def _ranks(sims, gt_idx):
    order = np.argsort(-sims, axis=1)
    return np.array([np.where(order[i] == gt_idx[i])[0][0] for i in range(len(gt_idx))])

def recall_at_k(sims, gt_idx, ks=(1, 5, 10)):
    rk = _ranks(sims, gt_idx)
    return {k: float((rk < k).mean() * 100) for k in ks}

def mean_ap(sims, gt_idx):
    rk = _ranks(sims, gt_idx)            # single positive per query -> AP = 1/(rank+1)
    return float(np.mean(1.0 / (rk + 1)) * 100)
