"""s2hy6 reproduction + ablation lab for the 4 open questions:
  Q1 round window 7->10/15/20    Q2 L5 override for '>=2 sources see GT'
  Q3 verifier-locked diagnosis   Q4 AWQ-vs-fp16 logprob equivalence
CPU-only, reuses all existing caches. Run from SSDC-luan/submissions/.
"""
import json, sys, os, collections
import numpy as np
sys.path.insert(0, '..')
from eval_submission import score

gt = json.load(open('gt.json'))
# pool overridable so the same stage-2 stack can rerank a different stage-1 base (e.g. the
# compliant LambdaMART pool) via SSDC_POOL=submissions/<run>/scores.json
scores = json.load(open(os.environ.get('SSDC_POOL', 'fuse_gated_joint128_blank/scores.json')))
awq = json.load(open('s2cache_logprob_gr15/verify_logprob.json'))['sem']
fp  = json.load(open('s2cache_logprob_fp16_gr15/verify_logprob.json'))['sem']
ivl = json.load(open('s2cache_internvl35/verify_logprob.json'))['sem']
smo = json.load(open('s2cache_smolvlm2/verify_logprob.json'))['sem']
deep= json.load(open('s2cache_logprob_deep/verify_logprob.json'))['sem']

def load_cache(name):
    vc = json.load(open(f'{name}/verify_cache.json'))
    parent = json.load(open(vc['run'] + '/submission.json'))
    return {q: {img: s for img, s in zip(parent.get(q, []), sc)} for q, sc in vc['sem'].items()}
old = {}
for c in [load_cache('s2cache_w03_r20'), load_cache('s2cache_w03_r10_fp16')]:
    for q, m in c.items(): old.setdefault(q, {}).update(m)

qids = list(scores.keys())
top1 = np.array([scores[q][0][1] for q in qids])
nrm = (top1 - top1.min()) / (top1.max() - top1.min() + 1e-9)
norm1 = {q: float(n) for q, n in zip(qids, nrm)}
GT = {q: gt[q]['gt_token'] + '.jpg' for q in gt}


def rank_q(q, wi=0.8, ws=0.3, rnd=7, drop_awq=False):
    xi, lam, a1, a2, g = 0.85, 0.5, 0.8, 0.2, 0.25
    if drop_awq:                      # Q4: fold AWQ weight into fp16 (single Qwen logprob)
        a1, a2 = 0.0, 1.0
    pool = [n for n, _ in scores[q]]
    struct = {n: s for n, s in scores[q]}
    mo, ma, mf = old.get(q, {}), awq.get(q, {}), fp.get(q, {})
    srcs = [(wi, ivl.get(q, {})), (ws, smo.get(q, {}))]
    if norm1[q] <= xi and (mo or ma or mf):
        seg = pool[:min(rnd, len(pool))]
        sv = np.array([struct[n] for n in seg]); sn = (sv - sv.min()) / (sv.max() - sv.min() + 1e-9)
        def qn(m):
            pv = [m.get(n) for n in seg]; k = [x for x in pv if x is not None]
            if k and max(k) > min(k):
                lo_, hi_ = min(k), max(k)
                return [((x-lo_)/(hi_-lo_)) if x is not None else None for x in pv]
            return [None]*len(seg)
        normed = [(w, qn(m)) for w, m in srcs]
        Z = 1.0 + a1 + a2 + sum(w for w, _ in srcs)
        sem = np.zeros(len(seg))
        for i, n in enumerate(seg):
            b = mo.get(n); pa = ma.get(n); pf = mf.get(n)
            if b is None: b = (pf if pf is not None else (pa if pa is not None else sn[i]*0.5))
            t = b + a1*((pa**g) if pa is not None else 0.5*b) + a2*((pf**g) if pf is not None else 0.5*b)
            for w, nv in normed:
                t += w * (nv[i] if nv[i] is not None else 0.5*b)
            sem[i] = t / Z
        fin = lam*sn + (1-lam)*sem
        order = [seg[i] for i in np.argsort(-fin)]
        return order + [n for n in pool if n not in set(order)]
    return pool


def apply_l4(sub0):
    out = {}
    for q in qids:
        lst = list(sub0[q]); dq = deep.get(q, {})
        if dq:
            good = []
            for img, p in dq.items():
                if p < 0.7 or img in lst[:10]: continue
                pf_ = fp.get(q, {}).get(img); b_ = old.get(q, {}).get(img)
                conf = (pf_ is not None and pf_ >= 0.5) or (b_ == 1.0)
                good.append((p + (0.5 if conf else 0), img))
            good.sort(reverse=True)
            if good:
                img = good[0][1]
                if img in lst: lst.remove(img)
                lst.insert(0, img)
        out[q] = lst
    return out


def apply_l1(sub0):
    out = {}
    for q in qids:
        lst = list(sub0[q]); t1 = lst[0]
        ma, mf, mo, mi = awq.get(q, {}), fp.get(q, {}), old.get(q, {}), ivl.get(q, {})
        pa1, pf1, pi1 = ma.get(t1), mf.get(t1), mi.get(t1)
        bj, bs_ = None, -1
        for j in lst[1:5]:
            paj, pfj, pij = ma.get(j), mf.get(j), mi.get(j)
            if None in (paj, pfj, pa1, pf1): continue
            ok = paj >= pa1+0.2 and pfj >= pf1+0.2 and mo.get(j) == 1.0
            ok = ok or (pij is not None and pi1 is not None and pij >= pi1+0.3
                        and paj >= pa1+0.1 and mo.get(j) == 1.0)
            if ok and paj+pfj > bs_: bj, bs_ = j, paj+pfj
        if bj: lst.remove(bj); lst.insert(0, bj)
        out[q] = lst
    return out


def apply_l5(sub0, mode='ivl_smo', dmarg=0.15, need=2):
    """Q2: rescue when the two DIVERSE-family sources (ivl, smo) minmax-prefer a
    lower-ranked candidate over top-1 by margin, WITHOUT requiring binary Yes
    (binary is saturated in this regime). Optionally also count awq/fp16 to reach `need`."""
    out = {}
    fired = 0
    for q in qids:
        lst = list(sub0[q]); t1 = lst[0]
        def mm(cache):                # per-query minmax over the top-5 window
            seg = lst[:5]; pv = [cache.get(q, {}).get(n) for n in seg]
            k = [x for x in pv if x is not None]
            if not k or max(k) <= min(k): return {}
            lo_, hi_ = min(k), max(k)
            return {n: (v-lo_)/(hi_-lo_) for n, v in zip(seg, pv) if v is not None}
        mi, ms = mm(ivl), mm(smo)
        ma, mf = mm(awq), mm(fp)
        cand = {}
        for j in lst[1:5]:
            votes = 0; strength = 0.0
            for cm in ([mi, ms] if mode == 'ivl_smo' else [mi, ms, ma, mf]):
                if j in cm and t1 in cm and cm[j] >= cm[t1] + dmarg:
                    votes += 1; strength += cm[j] - cm[t1]
            if votes >= need: cand[j] = strength
        if cand:
            j = max(cand, key=cand.get)
            lst.remove(j); lst.insert(0, j); fired += 1
        out[q] = lst
    return out, fired


def full_pipe(wi=0.8, ws=0.3, rnd=7, drop_awq=False, l5=None):
    base = {q: rank_q(q, wi, ws, rnd, drop_awq) for q in qids}
    x = apply_l1(apply_l4(base))
    fired = 0
    if l5 is not None:
        x, fired = apply_l5(x, **l5)
    return {q: x[q][:10] for q in qids}, fired


def r1(sub): return score(sub, gt)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('exp', nargs='?', default='all')
    a = ap.parse_args()
    base_sub, _ = full_pipe()
    m0 = r1(base_sub)
    print(f"== s2hy6 reproduced: R@1={m0['R@1']} R@5={m0['R@5']} R@10={m0['R@10']} mAP={m0['mAP']}\n")
