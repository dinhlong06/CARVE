"""Hyperparameter sweep for LambdaMART 3-encoder fusion, COMPLIANT:
select on a synthetic dev split (val-hard scenes), never on gt.json. Then retrain the winner
on all synthetic and apply to localeval (real) as the transfer diagnostic.

  python3 tools/lambdamart_sweep.py
"""
import os, sys, json, itertools
import numpy as np
sys.path.insert(0, '.')
from eval_submission import score as le_score
import lightgbm as lgb

CMP = '/dev/shm/cmp_vh_scores/cmp_scores.json'
SYN = {'vith': '/dev/shm/synvith', 'sig': '/dev/shm/synsig'}
LE = {'cmp': 'submissions/joint_v1_ep1_p128_blank/scores.json',
      'vith': '/dev/shm/levith', 'sig': '/dev/shm/lesig'}
QEXT = 'data/synthetic_eval/ext_inputs/queries_ext.jsonl'
GT = 'submissions/gt.json'


def load(d):
    sims = np.load(os.path.join(d, 'sims_ext.npy'))
    idx = json.load(open(os.path.join(d, 'index.json')))
    return sims, {os.path.splitext(n)[0]: i for i, n in enumerate(idx['names'])}


def ranks(x):
    o = np.argsort(-x); r = np.empty_like(o); r[o] = np.arange(len(x)); return r.astype(np.float32)


def feat_row(itm, cand, vrow, vcol, srow, scol):
    vs = np.array([vrow[vcol[c]] if c in vcol else -1. for c in cand], dtype=np.float32)
    ss = np.array([srow[scol[c]] if c in scol else -1. for c in cand], dtype=np.float32)
    return np.stack([itm, ranks(itm), vs, ranks(vs), ss, ranks(ss)], axis=1)


def main():
    cmp = json.load(open(CMP)); Q = len(cmp['queries'])
    sv, svc = load(SYN['vith']); ss, ssc = load(SYN['sig'])
    print(f"### building synthetic 3-enc features ({Q} q)...", flush=True)
    rows = []  # (scene, F, labels, cand, gid)
    for i in range(Q):
        gid = str(cmp['queries'][i]['image_id']); cand = cmp['gallery_ids'][i]
        itm = np.asarray(cmp['itm'][i], dtype=np.float32)
        F = feat_row(itm, cand, np.asarray(sv[i]), svc, np.asarray(ss[i]), ssc)
        lab = np.array([1 if c == gid else 0 for c in cand], dtype=np.int32)
        rows.append((gid.split('_')[0], F, lab, cand, gid))

    scenes = sorted({r[0] for r in rows})
    rng = np.random.RandomState(20260622); rng.shuffle(scenes)
    n = len(scenes); tr_s = set(scenes[:int(n*.6)]); dev_s = set(scenes[int(n*.6):int(n*.8)])
    # val_s = rest (held-out synthetic, honest final check)
    val_s = set(scenes[int(n*.8):])

    def pack(sel):
        X, Y, G, meta = [], [], [], []
        for scn, F, lab, cand, gid in rows:
            if scn in sel: X.append(F); Y.append(lab); G.append(len(cand)); meta.append((cand, gid))
        return np.concatenate(X), np.concatenate(Y), G, meta
    Xtr, Ytr, Gtr, _ = pack(tr_s)
    Xdev, Ydev, Gdev, dev_meta = pack(dev_s)
    Xval, Yval, Gval, val_meta = pack(val_s)
    print(f"### scenes {n} -> train {len(tr_s)} dev {len(dev_s)} val {len(val_s)}", flush=True)

    def rk(model, X, meta, G):
        h1 = h5 = h10 = 0; ap = 0.; off = 0
        for (cand, gid), g in zip(meta, G):
            sc = model.predict(X[off:off+g]); off += g
            order = np.argsort(-sc)
            pos = next((r for r, o in enumerate(order) if cand[o] == gid), 999)
            h1 += pos == 0; h5 += pos < 5; h10 += pos < 10
            ap += 1./(pos+1) if pos < 999 else 0
        m = len(meta); return 100*h1/m, 100*h5/m, 100*h10/m, 100*ap/m

    # ---- sweep ----
    grid = dict(num_leaves=[15, 31, 63], learning_rate=[0.03, 0.05, 0.1],
                min_data_in_leaf=[20, 50, 100], lambdarank_truncation_level=[10, 20])
    keys = list(grid); combos = list(itertools.product(*grid.values()))
    dtr = lgb.Dataset(Xtr, label=Ytr, group=Gtr)
    ddev = lgb.Dataset(Xdev, label=Ydev, group=Gdev, reference=dtr)
    results = []
    for ci, vals in enumerate(combos):
        p = dict(zip(keys, vals)); p.update(objective='lambdarank', metric='ndcg',
                  ndcg_eval_at=[1, 10], verbose=-1, num_threads=6)
        m = lgb.train(p, dtr, num_boost_round=400, valid_sets=[ddev],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        dev = rk(m, Xdev, dev_meta, Gdev)   # (R@1,R@5,R@10,mAP) on dev
        results.append((dev[0], dev[3], m.best_iteration, p, dev))
        if ci % 12 == 0: print(f"  [{ci+1}/{len(combos)}] dev R@1={dev[0]:.2f} mAP={dev[3]:.2f} {p}", flush=True)
    # select by dev mAP (more discriminative than saturated R@1)
    results.sort(key=lambda r: -r[1])
    best = results[0]
    print(f"\n### BEST by dev-mAP: {best[3]} (iters={best[2]}) dev R@1={best[4][0]:.2f} R@10={best[4][2]:.2f} mAP={best[1]:.2f}")
    print(f"### top-5 dev configs:")
    for r in results[:5]:
        print(f"   dev R@1={r[4][0]:.2f} R@10={r[4][2]:.2f} mAP={r[1]:.2f} it={r[2]}  {r[3]}")

    # ---- retrain best on ALL synthetic, apply to localeval ----
    Xall, Yall, Gall, _ = pack(tr_s | dev_s | val_s)
    bp = dict(best[3]); bp.update(objective='lambdarank', metric='ndcg', verbose=-1, num_threads=6)
    model = lgb.train(bp, lgb.Dataset(Xall, label=Yall, group=Gall), num_boost_round=best[2] or 200)
    # also a default-param model for comparison
    dp = dict(objective='lambdarank', num_leaves=31, learning_rate=0.05, min_data_in_leaf=50,
              lambdarank_truncation_level=20, verbose=-1, num_threads=6)
    model_def = lgb.train(dp, lgb.Dataset(Xall, label=Yall, group=Gall), num_boost_round=200)

    # localeval features + apply
    le = json.load(open(LE['cmp'])); lv, lvc = load(LE['vith']); ls, lsc = load(LE['sig'])
    levidx = json.load(open(os.path.join(LE['vith'], 'index.json')))['qidx']
    lrow = {q: i for i, q in enumerate(levidx)}
    gt = json.load(open(GT)); qids = list(le.keys())

    def le_apply(model, dump=None):
        sub = {}; pool = {}
        for q in qids:
            cand = [n.split('.')[0] for n, _ in le[q]]
            cmp_s = np.array([s for _, s in le[q]], dtype=np.float32); i = lrow[q]
            F = feat_row(cmp_s, cand, np.asarray(lv[i]), lvc, np.asarray(ls[i]), lsc)
            sc = model.predict(F); order = np.argsort(-sc)
            sub[q] = [cand[j]+'.jpg' for j in order][:10]
            if dump is not None: pool[q] = [[cand[j]+'.jpg', float(sc[j])] for j in order]
        if dump is not None:
            os.makedirs(os.path.dirname(dump), exist_ok=True); json.dump(pool, open(dump, 'w'))
        return le_score(sub, gt)

    # ---- BẢNG: mỗi config top val-hard -> retrain toàn synthetic -> gt_local ----
    print("\n### val-hard-dev  ->  gt_local (mỗi config tối ưu trên val-hard):")
    print(f"{'cfg':<48} {'devR@1':>7} {'devmAP':>7} | {'geR@1':>6} {'geR@10':>7} {'gemAP':>6}")
    md = le_apply(model_def)
    print(f"{'DEFAULT(nl31,lr.05,md50,tr20,200it)':<48} {'-':>7} {'-':>7} | "
          f"{md['R@1']:6.2f} {md['R@10']:7.2f} {md['mAP']:6.2f}")
    for rk_ in results[:6]:
        p = dict(rk_[3]); it = rk_[2] or 200
        p2 = dict(p); p2.update(objective='lambdarank', metric='ndcg', verbose=-1, num_threads=6)
        m = lgb.train(p2, lgb.Dataset(Xall, label=Yall, group=Gall), num_boost_round=it)
        ge = le_apply(m)
        tag = f"nl{p['num_leaves']},lr{p['learning_rate']},md{p['min_data_in_leaf']},tr{p['lambdarank_truncation_level']},{it}it"
        print(f"{tag:<48} {rk_[4][0]:7.2f} {rk_[1]:7.2f} | {ge['R@1']:6.2f} {ge['R@10']:7.2f} {ge['mAP']:6.2f}", flush=True)


if __name__ == '__main__':
    main()
