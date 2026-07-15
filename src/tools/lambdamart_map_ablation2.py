"""mAP ablation v2 — COMPLIANT feature-set selection.
Feature engineering (base -> rich -> rich2) is selected on a SYNTHETIC held-out scene split
(dev), NEVER on gt_local. Localeval is reported ONLY as post-hoc verification (does the
synthetic-chosen design also transfer?), not for selection. Hyperparameters are fixed robust
defaults (trunc20, 200 rounds, num_leaves31) — not tuned on either set.

  python3 tools/lambdamart_map_ablation2.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, '.')
from eval_submission import score as le_score
import lightgbm as lgb

CMP = '/dev/shm/cmp_vh_scores/cmp_scores.json'
SYN = {'vith': '/dev/shm/synvith/sims_ext.npy', 'sig': '/dev/shm/synsig/sims_ext.npy'}
LE = {'cmp': 'submissions/joint_v1_ep1_p128_blank/scores.json',
      'vith': '/dev/shm/levith/sims_ext.npy', 'sig': '/dev/shm/lesig/sims_ext.npy'}
GT = 'submissions/gt.json'
DEFAULT = dict(objective='lambdarank', metric='ndcg', num_leaves=31, learning_rate=0.05,
               min_data_in_leaf=50, lambdarank_truncation_level=20, verbose=-1, num_threads=6)


def load_sims(path):
    d = os.path.dirname(path); sims = np.load(path)
    idx = json.load(open(os.path.join(d, 'index.json')))
    return sims, {os.path.splitext(n)[0]: i for i, n in enumerate(idx['names'])}


def ranks(x):
    o = np.argsort(-x); r = np.empty_like(o); r[o] = np.arange(len(x)); return r.astype(np.float32)


def znorm(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def feats(cmp_s, vs, ss, level):
    rc, rv, rs = ranks(cmp_s), ranks(vs), ranks(ss)
    cols = [cmp_s, rc, vs, rv, ss, rs]
    if level >= 1:
        zc, zv, zs = znorm(cmp_s), znorm(vs), znorm(ss)
        cols += [zc, zv, zs, (zc + zv + zs) / 3.0,
                 cmp_s - cmp_s.max(), vs - vs.max(), ss - ss.max(), zc * zv, zc * zs, zv * zs]
    if level >= 2:
        R = np.stack([rc, rv, rs], axis=1); Z = np.stack([znorm(cmp_s), znorm(vs), znorm(ss)], axis=1)
        cols += [1. / (60 + rc), 1. / (60 + rv), 1. / (60 + rs),
                 R.mean(1), R.min(1), R.max(1) - R.min(1),
                 (R < 5).sum(1).astype(np.float32), (R < 10).sum(1).astype(np.float32), Z.std(1)]
    return np.stack(cols, axis=1)


def build_syn(level):
    cmp = json.load(open(CMP)); vith, vcol = load_sims(SYN['vith']); sig, scol = load_sims(SYN['sig'])
    rows = []
    for i in range(len(cmp['queries'])):
        gid = str(cmp['queries'][i]['image_id']); cand = cmp['gallery_ids'][i]
        cmp_s = np.asarray(cmp['itm'][i], np.float32)
        vrow = np.asarray(vith[i]); vs = np.array([vrow[vcol[c]] if c in vcol else -1. for c in cand], np.float32)
        srow = np.asarray(sig[i]); ss = np.array([srow[scol[c]] if c in scol else -1. for c in cand], np.float32)
        lab = np.array([1 if c == gid else 0 for c in cand], np.int32)
        rows.append((gid.split('_')[0], feats(cmp_s, vs, ss, level), lab, cand, gid))
    return rows


def le_build(level):
    le = json.load(open(LE['cmp'])); lev, lvc = load_sims(LE['vith']); les, lsc = load_sims(LE['sig'])
    lrow = {q: i for i, q in enumerate(json.load(open(os.path.join(os.path.dirname(LE['vith']), 'index.json')))['qidx'])}
    out = {}
    for q in le:
        cand = [n.split('.')[0] for n, _ in le[q]]
        cmp_s = np.array([s for _, s in le[q]], np.float32); i = lrow[q]
        vrow = np.asarray(lev[i]); vs = np.array([vrow[lvc[c]] if c in lvc else -1. for c in cand], np.float32)
        srow = np.asarray(les[i]); ss = np.array([srow[lsc[c]] if c in lsc else -1. for c in cand], np.float32)
        out[q] = ([c + '.jpg' for c in cand], feats(cmp_s, vs, ss, level))
    return out, json.load(open(GT))


def syn_dev_score(model, dev_rows):
    """R@1/mAP on synthetic held-out scenes (single relevant per query -> mAP == MRR)."""
    h1 = ap = 0; n = 0
    for scn, F, lab, cand, gid in dev_rows:
        sc = model.predict(F); order = np.argsort(-sc)
        pos = next((r for r, o in enumerate(order) if cand[o] == gid), 999)
        h1 += pos == 0; ap += 1. / (pos + 1) if pos < 999 else 0; n += 1
    return 100 * h1 / n, 100 * ap / n


def le_verify(model, le_data, gt, dump=None):
    sub = {}; pool = {}
    for q, (cand, Xq) in le_data.items():
        lm = model.predict(Xq); o = np.argsort(-lm)
        sub[q] = [cand[j] for j in o][:10]
        if dump is not None: pool[q] = [[cand[j], float(lm[j])] for j in o]
    m = le_score(sub, gt)
    if dump is not None:
        os.makedirs(os.path.dirname(dump), exist_ok=True); json.dump(pool, open(dump, 'w'))
    return m


if __name__ == '__main__':
    # scene-split synthetic for compliant selection
    rows_by_level = {L: build_syn(L) for L in (0, 1, 2)}
    scenes = sorted({r[0] for r in rows_by_level[0]})
    rng = np.random.RandomState(20260622); rng.shuffle(scenes)
    dev_s = set(scenes[:int(len(scenes) * 0.2)])
    le_by_level = {L: le_build(L) for L in (0, 1, 2)}
    gt = le_by_level[0][1]

    print("### feature-set selection on SYNTHETIC dev (compliant) | localeval = verify only:")
    print(f"{'level':32s} {'synDEV R@1':>10} {'synDEV mAP':>10} | {'LE R@1':>7} {'LE mAP':>7}")
    results = []
    for L, name in [(0, 'base (6 feat)'), (1, 'rich (16 feat)'), (2, 'rich2 (25 feat)')]:
        rows = rows_by_level[L]
        tr = [r for r in rows if r[0] not in dev_s]; dv = [r for r in rows if r[0] in dev_s]
        X = np.concatenate([r[1] for r in tr]); Y = np.concatenate([r[2] for r in tr]); G = [len(r[3]) for r in tr]
        model = lgb.train(DEFAULT, lgb.Dataset(X, label=Y, group=G), num_boost_round=200)
        d1, dmap = syn_dev_score(model, dv)
        le_data, _ = le_by_level[L]; le_m = le_verify(model, le_data, gt)
        print(f"{name:32s} {d1:10.2f} {dmap:10.2f} | {le_m['R@1']:7.2f} {le_m['mAP']:7.2f}", flush=True)
        results.append((L, name, dmap, d1))

    # COMPLIANT choice = best synthetic-dev mAP; retrain on ALL synthetic, dump, verify
    results.sort(key=lambda r: -r[2])
    bestL, bestname = results[0][0], results[0][1]
    print(f"\n### synthetic-dev picks: {bestname} (level {bestL})  <- adopt this (NOT gt-selected)")
    rows = rows_by_level[bestL]
    X = np.concatenate([r[1] for r in rows]); Y = np.concatenate([r[2] for r in rows]); G = [len(r[3]) for r in rows]
    model = lgb.train(DEFAULT, lgb.Dataset(X, label=Y, group=G), num_boost_round=200)
    le_data, _ = le_by_level[bestL]
    m = le_verify(model, le_data, gt, dump='submissions/lambdamart_pool_3enc_map2/scores.json')
    print(f"### VERIFY on localeval (full synthetic train, default params): "
          f"R@1={m['R@1']:.2f} R@5={m['R@5']:.2f} R@10={m['R@10']:.2f} mAP={m['mAP']:.2f}")
    print("### dumped -> submissions/lambdamart_pool_3enc_map2/scores.json")
