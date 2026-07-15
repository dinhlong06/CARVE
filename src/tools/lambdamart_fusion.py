"""Learned fusion (LightGBM LambdaMART) over CMP + ViT-H + SigLIP on the synthetic val-hard
set. Per (query, candidate-in-CMP-top128) features = {itc, itm, ranks, ViT-H sim/rank,
SigLIP sim/rank}; label = candidate == GT image_id. Split the 75 val-hard scenes into
train/val (no scene leakage) -> train lambdarank -> report R@1/5/10 vs CMP-itm and RRF.
All synthetic => Track-4 compliant. Applying to localeval (real) is a separate diagnostic.

  python3 tools/lambdamart_fusion.py --siglip submissions/ext_vh_siglip/features/sims_ext.npy
"""
import os, sys, json, argparse
import numpy as np


def load_sims(path):
    d = os.path.dirname(path)
    sims = np.load(path, mmap_mode='r')
    idx = json.load(open(os.path.join(d, 'index.json')))
    col = {os.path.splitext(n)[0]: i for i, n in enumerate(idx['names'])}
    return sims, col, idx['qidx']


def ranks(x):
    """dense rank (0=best) of each element within its row, descending value."""
    order = np.argsort(-x)
    r = np.empty_like(order); r[order] = np.arange(len(x))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cmp', default='/dev/shm/cmp_vh_scores/cmp_scores.json')
    ap.add_argument('--vith', default='submissions/ext_vh_vith/features/sims_ext.npy')
    ap.add_argument('--siglip', default='submissions/ext_vh_siglip/features/sims_ext.npy')
    ap.add_argument('--queries', default='data/synthetic_eval/ext_inputs/queries_ext.jsonl')
    ap.add_argument('--val_frac', type=float, default=0.2, help="fraction of scenes held out")
    ap.add_argument('--out', default='submissions/lambdamart_vh.txt')
    a = ap.parse_args()

    cmp = json.load(open(a.cmp))
    K = cmp['K']; Q = len(cmp['queries'])
    qrows = [json.loads(l) for l in open(a.queries)]
    assert len(qrows) == Q, f"queries_ext {len(qrows)} != cmp {Q}"
    # position alignment check: cmp queries[i].image_id must equal queries_ext[i].image_id
    for i in range(0, Q, 2000):
        assert str(cmp['queries'][i]['image_id']) == str(qrows[i]['image_id']), f"misalign at {i}"

    vith, vcol, _ = load_sims(a.vith)
    have_sig = os.path.exists(a.siglip)
    if have_sig:
        sig, scol, _ = load_sims(a.siglip)
    print(f"### {Q} queries, K={K}, siglip={'yes' if have_sig else 'NO'}", flush=True)

    # scene split (prefix before first '_')
    scenes = sorted({q['image_id'].split('_')[0] for q in cmp['queries']})
    rng = np.random.RandomState(20260622); rng.shuffle(scenes)
    nval = int(len(scenes) * a.val_frac)
    val_scenes = set(scenes[:nval])
    print(f"### {len(scenes)} scenes -> {len(scenes)-nval} train / {nval} val", flush=True)

    # build feature rows
    feats_tr, lab_tr, grp_tr = [], [], []
    feats_va, lab_va, grp_va = [], [], []
    va_meta = []   # (query_pos, cand_gallery_ids, gt) for val eval
    for i in range(Q):
        gid = str(cmp['queries'][i]['image_id'])
        cand = cmp['gallery_ids'][i]              # top-K gallery ids
        itc = np.asarray(cmp['itc'][i], dtype=np.float32)
        itm = np.asarray(cmp['itm'][i], dtype=np.float32)
        itc_r = ranks(itc).astype(np.float32)
        itm_r = ranks(itm).astype(np.float32)
        vs = np.array([vith[i, vcol[c]] if c in vcol else -1.0 for c in cand], dtype=np.float32)
        vs_r = ranks(vs).astype(np.float32)
        if have_sig:
            ss = np.array([sig[i, scol[c]] if c in scol else -1.0 for c in cand], dtype=np.float32)
            ss_r = ranks(ss).astype(np.float32)
            F = np.stack([itc, itm, itc_r, itm_r, vs, vs_r, ss, ss_r], axis=1)
        else:
            F = np.stack([itc, itm, itc_r, itm_r, vs, vs_r], axis=1)
        lab = np.array([1 if c == gid else 0 for c in cand], dtype=np.int32)
        scene = gid.split('_')[0]
        if scene in val_scenes:
            feats_va.append(F); lab_va.append(lab); grp_va.append(len(cand))
            va_meta.append((cand, gid))
        else:
            feats_tr.append(F); lab_tr.append(lab); grp_tr.append(len(cand))

    Xtr = np.concatenate(feats_tr); Ytr = np.concatenate(lab_tr)
    Xva = np.concatenate(feats_va); Yva = np.concatenate(lab_va)
    print(f"### train {Xtr.shape} ({len(grp_tr)} q), val {Xva.shape} ({len(grp_va)} q)", flush=True)

    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, label=Ytr, group=grp_tr)
    dva = lgb.Dataset(Xva, label=Yva, group=grp_va, reference=dtr)
    params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 5, 10],
                  learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                  lambdarank_truncation_level=20, verbose=-1)
    model = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])

    # eval R@k on val: LambdaMART vs CMP-itm vs RRF(itc,itm,vith,siglip)
    def recall(scorer):
        h1 = h5 = h10 = 0
        off = 0
        for (cand, gid), g in zip(va_meta, grp_va):
            sc = scorer(Xva[off:off + g], cand); off += g
            order = np.argsort(-sc)
            pos = next((r for r, o in enumerate(order) if cand[o] == gid), 999)
            h1 += pos == 0; h5 += pos < 5; h10 += pos < 10
        n = len(va_meta)
        return 100 * h1 / n, 100 * h5 / n, 100 * h10 / n

    def rrf(X, cand, k=60):
        # ranks are cols 2(itc_r),3(itm_r),5(vith_r),7(siglip_r if present)
        cols = [2, 3, 5] + ([7] if have_sig else [])
        return sum(1.0 / (k + X[:, c]) for c in cols)

    lm = lambda X, c: model.predict(X)
    itm_only = lambda X, c: X[:, 1]
    print("\n### VAL R@1/R@5/R@10 (synthetic held-out scenes):")
    for name, fn in [('CMP-itm', itm_only), ('RRF', rrf), ('LambdaMART', lm)]:
        r = recall(fn)
        print(f"  {name:12s} R@1={r[0]:.2f} R@5={r[1]:.2f} R@10={r[2]:.2f}")
    imp = dict(zip(['itc', 'itm', 'itc_r', 'itm_r', 'vith', 'vith_r', 'sig', 'sig_r'][:Xtr.shape[1]],
                   model.feature_importance().tolist()))
    print("### feature importance:", imp)
    model.save_model(a.out.replace('.txt', '.lgb'))
    print(f"### saved model -> {a.out.replace('.txt', '.lgb')}")


if __name__ == '__main__':
    main()
