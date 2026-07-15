"""Transfer test: train LambdaMART fusion on SYNTHETIC val-hard, apply to LOCALEVAL (real).
Unified feature schema available on BOTH sets (no fresh localeval CMP dump needed):
  cmp_score, cmp_rank, vith_sim, vith_rank [, sig_sim, sig_rank]
On synthetic, cmp_score = ITM (full stage-1 rerank score). On localeval, cmp_score =
joint_v1_ep1_p128_blank/scores.json value (same itm-reranked CMP signal).
Train synthetic (scene-split, compliant) -> apply localeval -> R@1 vs CMP-only vs gated-RRF.

  python3 tools/lambdamart_transfer.py            # cmp+vith
  python3 tools/lambdamart_transfer.py --with_sig # add SigLIP (needs val-hard sig sims)
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, '.')
from eval_submission import score as le_score


def load_sims(path):
    d = os.path.dirname(path)
    # full RAM load (NOT mmap): random per-element mmap access on a 2GB NFS-backed file
    # thrashes under host memory/CPU contention; a one-time sequential read is far faster
    sims = np.load(path)
    idx = json.load(open(os.path.join(d, 'index.json')))
    col = {os.path.splitext(n)[0]: i for i, n in enumerate(idx['names'])}
    print(f"### loaded sims {sims.shape} into RAM from {path}", flush=True)
    return sims, col


def ranks(x):
    order = np.argsort(-x); r = np.empty_like(order); r[order] = np.arange(len(x))
    return r.astype(np.float32)


def build_syn(cmp, vith, vcol, sig, scol, with_sig):
    Q = len(cmp['queries'])
    rows, meta = [], []
    for i in range(Q):
        gid = str(cmp['queries'][i]['image_id']); cand = cmp['gallery_ids'][i]
        itm = np.asarray(cmp['itm'][i], dtype=np.float32)
        vrow = np.asarray(vith[i])            # read the whole row once (fast) then gather cols
        vs = np.array([vrow[vcol[c]] if c in vcol else -1.0 for c in cand], dtype=np.float32)
        cols = [itm, ranks(itm), vs, ranks(vs)]
        if with_sig:
            srow = np.asarray(sig[i])
            ss = np.array([srow[scol[c]] if c in scol else -1.0 for c in cand], dtype=np.float32)
            cols += [ss, ranks(ss)]
        F = np.stack(cols, axis=1)
        lab = np.array([1 if c == gid else 0 for c in cand], dtype=np.int32)
        rows.append((gid.split('_')[0], F, lab, cand, gid))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cmp', default='/dev/shm/cmp_vh_scores/cmp_scores.json')
    ap.add_argument('--syn_vith', default='/dev/shm/synvith/sims_ext.npy')
    ap.add_argument('--syn_sig', default='submissions/ext_vh_siglip/features/sims_ext.npy')
    ap.add_argument('--le_cmp', default='submissions/joint_v1_ep1_p128_blank/scores.json')
    ap.add_argument('--le_vith', default='/dev/shm/levith/sims_ext.npy')
    ap.add_argument('--le_sig', default='submissions/ext_siglip/features/sims_ext.npy')
    ap.add_argument('--gt', default='submissions/gt.json')
    ap.add_argument('--with_sig', action='store_true')
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--dump_pool', default=None, help="save LambdaMART-scored localeval pool (scores.json fmt)")
    a = ap.parse_args()
    import lightgbm as lgb

    # ---- SYNTHETIC: build + scene-split + train ----
    cmp = json.load(open(a.cmp))
    vith, vcol = load_sims(a.syn_vith)
    sig, scol = (load_sims(a.syn_sig) if a.with_sig else (None, None))
    rows = build_syn(cmp, vith, vcol, sig, scol, a.with_sig)
    scenes = sorted({r[0] for r in rows}); rng = np.random.RandomState(20260622); rng.shuffle(scenes)
    val_scenes = set(scenes[:int(len(scenes) * a.val_frac)])
    Xtr, Ytr, Gtr = [], [], []
    for scn, F, lab, cand, gid in rows:
        if scn in val_scenes: continue
        Xtr.append(F); Ytr.append(lab); Gtr.append(len(cand))
    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
    print(f"### syn train {Xtr.shape} ({len(Gtr)} q), with_sig={a.with_sig}", flush=True)
    dtr = lgb.Dataset(Xtr, label=Ytr, group=Gtr)
    # cap threads: host is CPU-oversubscribed (load ~236 on 40 cores); LightGBM's default
    # "all cores" spawns 40 threads that thrash against other users -> training crawls
    params = dict(objective='lambdarank', metric='ndcg', learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=50, lambdarank_truncation_level=20, verbose=-1, num_threads=6)
    model = lgb.train(params, dtr, num_boost_round=200)
    fn = ['cmp', 'cmp_r', 'vith', 'vith_r'] + (['sig', 'sig_r'] if a.with_sig else [])
    print("### feat importance:", dict(zip(fn, model.feature_importance().tolist())), flush=True)

    # ---- LOCALEVAL: apply, no gt in features ----
    le = json.load(open(a.le_cmp))            # {q: [[name, score]*128]}
    lev, lvcol = load_sims(a.le_vith)
    les, lscol = (load_sims(a.le_sig) if a.with_sig else (None, None))
    levidx = json.load(open(os.path.join(os.path.dirname(a.le_vith), 'index.json')))['qidx']
    lev_row = {q: i for i, q in enumerate(levidx)}
    gt = json.load(open(a.gt))
    qids = list(le.keys())

    def le_feats(q):
        cand = [n.split('.')[0] for n, _ in le[q]]     # gallery tokens
        cmp_s = np.array([s for _, s in le[q]], dtype=np.float32)
        i = lev_row[q]
        vrow = np.asarray(lev[i])
        vs = np.array([vrow[lvcol[c]] if c in lvcol else -1.0 for c in cand], dtype=np.float32)
        cols = [cmp_s, ranks(cmp_s), vs, ranks(vs)]
        if a.with_sig:
            srow = np.asarray(les[i])
            ss = np.array([srow[lscol[c]] if c in lscol else -1.0 for c in cand], dtype=np.float32)
            cols += [ss, ranks(ss)]
        return cand, np.stack(cols, axis=1)

    subs = {'CMP-only': {}, 'LambdaMART': {}, 'RRF': {}}
    lm_pool = {}     # {q: [[name.jpg, lm_score]*128]} for feeding the stage-2 stack
    for q in qids:
        cand, X = le_feats(q)
        lm = model.predict(X)
        subs['CMP-only'][q] = [cand[j] + '.jpg' for j in np.argsort(-X[:, 0])][:10]
        order = np.argsort(-lm)
        subs['LambdaMART'][q] = [cand[j] + '.jpg' for j in order][:10]
        lm_pool[q] = [[cand[j] + '.jpg', float(lm[j])] for j in order]
        rrf = 1.0 / (60 + X[:, 1]) + 1.0 / (60 + X[:, 3]) + (1.0 / (60 + X[:, 5]) if a.with_sig else 0)
        subs['RRF'][q] = [cand[j] + '.jpg' for j in np.argsort(-rrf)][:10]

    print("\n### LOCALEVAL (real perturbed) — transfer of synthetic-trained fusion:")
    for name, sub in subs.items():
        m = le_score(sub, gt)
        print(f"  {name:12s} R@1={m['R@1']:.2f} R@5={m['R@5']:.2f} R@10={m['R@10']:.2f} mAP={m['mAP']:.2f}")

    if a.dump_pool:
        os.makedirs(os.path.dirname(a.dump_pool), exist_ok=True)
        json.dump(lm_pool, open(a.dump_pool, 'w'))
        print(f"### dumped LambdaMART pool -> {a.dump_pool} ({len(lm_pool)} q)")


if __name__ == '__main__':
    main()
