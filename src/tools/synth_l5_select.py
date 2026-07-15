"""COMPLIANT selection of the s2hy7 L5 override params on the SYNTHETIC hard set.
Mirrors the default s2hy6 blend + L1 + L5 on the 500 synthetic-hard queries (their own
synthetic gt), sweeps (mode, dmarg, need) from an a-priori grid, and picks the best on
SYNTHETIC. gt_local is NEVER consulted here — the winner is then transferred to localeval by
a separate step. Run from SSDC-luan/submissions/.
"""
import json, os, numpy as np

pool = json.load(open('syn_hard_run/scores.json'))
gt   = json.load(open('syn_hard_run/gt_token.json'))      # {q: 'A_B'}
qwen = json.load(open('syn_hard_qwen/verify_logprob.json'))['sem']       # {q:{img:sem}}
ivl  = json.load(open('syn_hard_internvl/verify_logprob.json'))['sem']
smo  = json.load(open('syn_hard_smolvlm/verify_logprob.json'))['sem']
qids = list(pool.keys())
GT   = {q: gt[q] + '.jpg' for q in qids}

top1 = np.array([pool[q][0][1] for q in qids])
nrm  = (top1 - top1.min()) / (top1.max() - top1.min() + 1e-9)
norm1 = {q: float(n) for q, n in zip(qids, nrm)}


def rank_q(q, wi=0.8, ws=0.3, rnd=7):
    """Default s2hy6 blend. Synthetic has ONE Qwen cache -> use it as both awq & fp
    (they are the same model on localeval); no binary/old cache -> b falls back to Qwen."""
    xi, lam, a1, a2, g = 0.85, 0.5, 0.8, 0.2, 0.25
    names = [c[0] for c in pool[q]]; struct = {c[0]: c[1] for c in pool[q]}
    ma = qwen.get(q, {}); mf = ma                     # awq==fp==Qwen
    srcs = [(wi, ivl.get(q, {})), (ws, smo.get(q, {}))]
    if norm1[q] <= xi and ma:
        seg = names[:min(rnd, len(names))]
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
            pa = ma.get(n); pf = mf.get(n)
            b = pf if pf is not None else (pa if pa is not None else sn[i]*0.5)
            t = b + a1*((pa**g) if pa is not None else 0.5*b) + a2*((pf**g) if pf is not None else 0.5*b)
            for w, nv in normed:
                t += w * (nv[i] if nv[i] is not None else 0.5*b)
            sem[i] = t / Z
        fin = lam*sn + (1-lam)*sem
        order = [seg[i] for i in np.argsort(-fin)]
        return order + [n for n in names if n not in set(order)]
    return names


def apply_l1(sub0):
    out = {}
    for q in qids:
        lst = list(sub0[q]); t1 = lst[0]
        ma, mf, mi = qwen.get(q, {}), qwen.get(q, {}), ivl.get(q, {})
        pa1, pf1, pi1 = ma.get(t1), mf.get(t1), mi.get(t1)
        bj, bs_ = None, -1
        for j in lst[1:5]:
            paj, pfj, pij = ma.get(j), mf.get(j), mi.get(j)
            if None in (paj, pfj, pa1, pf1): continue
            ok = paj >= pa1+0.2 and pfj >= pf1+0.2               # no binary -> drop the '==1.0' gate
            ok = ok or (pij is not None and pi1 is not None and pij >= pi1+0.3 and paj >= pa1+0.1)
            if ok and paj+pfj > bs_: bj, bs_ = j, paj+pfj
        if bj: lst.remove(bj); lst.insert(0, bj)
        out[q] = lst
    return out


def apply_l5(sub0, mode='ivl_smo', dmarg=0.15, need=2):
    out = {}; fired = 0
    for q in qids:
        lst = list(sub0[q]); t1 = lst[0]
        def mm(cache):
            seg = lst[:5]; pv = [cache.get(q, {}).get(n) for n in seg]
            k = [x for x in pv if x is not None]
            if not k or max(k) <= min(k): return {}
            lo_, hi_ = min(k), max(k)
            return {n: (v-lo_)/(hi_-lo_) for n, v in zip(seg, pv) if v is not None}
        mi, ms, mq = mm(ivl), mm(smo), mm(qwen)
        cand = {}
        for j in lst[1:5]:
            votes = 0; strength = 0.0
            cms = [mi, ms] if mode == 'ivl_smo' else [mi, ms, mq]   # 'all' adds Qwen as 3rd
            for cm in cms:
                if j in cm and t1 in cm and cm[j] >= cm[t1] + dmarg:
                    votes += 1; strength += cm[j] - cm[t1]
            if votes >= need: cand[j] = strength
        if cand:
            j = max(cand, key=cand.get); lst.remove(j); lst.insert(0, j); fired += 1
        out[q] = lst
    return out, fired


def metrics(sub):
    """sub[q] = ordered name list. Single relevant per query -> mAP == MRR."""
    h1 = h5 = h10 = ap = 0
    for q in qids:
        names = [n.split('.')[0] for n in sub[q]]
        pos = names.index(gt[q]) if gt[q] in names else 999
        h1 += pos == 0; h5 += pos < 5; h10 += pos < 10
        ap += 1.0/(pos+1) if pos < 999 else 0
    n = len(qids)
    return dict(R1=100*h1/n, R5=100*h5/n, R10=100*h10/n, mAP=100*ap/n)


if __name__ == '__main__':
    base = {q: rank_q(q) for q in qids}
    base = apply_l1(base)
    m_base = metrics(base)
    print(f"synthetic pre-L5 (default blend + L1): R@1={m_base['R1']:.2f} mAP={m_base['mAP']:.2f}\n")

    grid = []
    for mode in ('ivl_smo', 'all'):
        needs = (1, 2) if mode == 'ivl_smo' else (2, 3)
        for need in needs:
            for dmarg in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
                x, fired = apply_l5(base, mode=mode, dmarg=dmarg, need=need)
                m = metrics(x)
                grid.append((m['R1'], m['mAP'], mode, need, dmarg, fired))

    print(f"{'mode':9s} {'need':>4} {'dmarg':>6} {'fired':>6} {'synR@1':>7} {'synmAP':>7}  d(R@1)")
    for r1, mp, mode, need, dmarg, fired in sorted(grid, key=lambda r: (-r[0], -r[1])):
        print(f"{mode:9s} {need:4d} {dmarg:6.2f} {fired:6d} {r1:7.2f} {mp:7.2f}  {r1-m_base['R1']:+.2f}")

    best = max(grid, key=lambda r: (r[0], r[1]))
    print(f"\n### SYNTHETIC-SELECTED L5 = mode={best[2]} need={best[3]} dmarg={best[4]:.2f} "
          f"(synR@1 {best[0]:.2f}, +{best[0]-m_base['R1']:.2f} over pre-L5)")
    json.dump({'mode': best[2], 'need': best[3], 'dmarg': best[4]},
              open('syn_hard_run/l5_selected.json', 'w'))
    print("### wrote syn_hard_run/l5_selected.json")
