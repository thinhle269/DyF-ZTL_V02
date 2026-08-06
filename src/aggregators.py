import copy
import torch
import numpy as np

# Robust aggregation baselines for the reviewer-requested SOTA comparison.
# All operate on lists of state_dicts from the same architecture and are given
# the oracle number of attackers f (best case for the baseline), so the
# comparison cannot be accused of using weakened baselines.
#
# References:
#   Krum / Multi-Krum: Blanchard et al., NeurIPS 2017
#   Coordinate-wise median / trimmed mean: Yin et al., ICML 2018
#   FLTrust: Cao et al., NDSS 2021

def _flatten(sd):
    return torch.cat([v.detach().float().reshape(-1) for v in sd.values()])

def _unflatten_like(vec, ref_sd):
    out, offset = {}, 0
    for k, v in ref_sd.items():
        n = v.numel()
        out[k] = vec[offset:offset + n].reshape(v.shape).to(v.dtype)
        offset += n
    return out

def krum(weights, f, multi=False):
    n = len(weights)
    f = int(max(0, min(f, (n - 3) // 2)))  # Krum requires n >= 2f+3
    flats = torch.stack([_flatten(w) for w in weights])
    d2 = torch.cdist(flats.unsqueeze(0), flats.unsqueeze(0)).squeeze(0) ** 2
    k = max(1, n - f - 2)
    scores = []
    for i in range(n):
        d = d2[i].clone()
        d[i] = float('inf')
        scores.append(torch.topk(d, k, largest=False).values.sum().item())
    order = np.argsort(scores)
    if multi:
        m = max(1, n - f)
        selected = [weights[i] for i in order[:m]]
        return average_weights(selected), list(order[:m])
    return copy.deepcopy(weights[order[0]]), [int(order[0])]

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        stacked = torch.stack([wi[key].float() for wi in w], dim=0)
        w_avg[key] = stacked.mean(dim=0).to(w[0][key].dtype)
    return w_avg

def coordinate_median(weights):
    out = copy.deepcopy(weights[0])
    for key in out.keys():
        stacked = torch.stack([wi[key].float() for wi in weights], dim=0)
        out[key] = stacked.median(dim=0).values.to(weights[0][key].dtype)
    return out

def trimmed_mean(weights, f):
    n = len(weights)
    k = int(max(0, min(f, (n - 1) // 2)))
    out = copy.deepcopy(weights[0])
    for key in out.keys():
        stacked = torch.stack([wi[key].float() for wi in weights], dim=0)
        if n - 2 * k < 1:
            out[key] = stacked.median(dim=0).values.to(weights[0][key].dtype)
        else:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            out[key] = sorted_vals[k:n - k].mean(dim=0).to(weights[0][key].dtype)
    return out

def fltrust(global_sd, client_sds, root_sd):
    g_flat = _flatten(global_sd)
    g0 = _flatten(root_sd) - g_flat
    g0_norm = g0.norm() + 1e-12

    total, weighted = 0.0, torch.zeros_like(g0)
    trust_scores = []
    for sd in client_sds:
        gi = _flatten(sd) - g_flat
        ts = torch.relu(torch.nn.functional.cosine_similarity(gi.unsqueeze(0), g0.unsqueeze(0))).item()
        trust_scores.append(ts)
        if ts > 0:
            gi_scaled = gi * (g0_norm / (gi.norm() + 1e-12))
            weighted += ts * gi_scaled
            total += ts

    if total == 0:
        return copy.deepcopy(global_sd), trust_scores
    new_flat = g_flat + weighted / total
    return _unflatten_like(new_flat, global_sd), trust_scores
