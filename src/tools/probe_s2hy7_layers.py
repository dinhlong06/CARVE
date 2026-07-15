"""Decompose the s2hy7 stack on the rich pool: how much R@1/mAP comes from the
DEFAULT s2hy6 blend (default function args) vs the gt-tuned L1/L4/L5 override layers.
CPU-only. Run from src/submissions/ with SSDC_POOL set to the rich pool.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(os.path.dirname(HERE), 'submissions')
pool = os.environ.get('SSDC_POOL', 'lambdamart_pool_3enc_rich/scores.json')
if pool.startswith('submissions/'):
    pool = pool[len('submissions/'):]
os.environ['SSDC_POOL'] = pool
os.chdir(SUB)
sys.path.insert(0, HERE)
import s2hy6_lab as L

def sc(x):
    m = L.r1({q: x[q][:10] for q in L.qids})
    return m['R@1'], m['R@5'], m['R@10'], m['mAP']

# stage-1 pool as-is (no stage-2 at all)
pool_only = {q: [n for n, _ in L.scores[q]] for q in L.qids}
# default s2hy6 blend, using the function's OWN default args
base = {q: L.rank_q(q, 0.8, 0.3, 7) for q in L.qids}
l1   = L.apply_l1(base)
l14  = L.apply_l1(L.apply_l4(base))
l145, fired = L.apply_l5(L.apply_l1(L.apply_l4(base)), mode='ivl_smo', dmarg=0.3, need=2)

for name, x in [("stage-1 rich pool only", pool_only),
                ("+ default s2hy6 blend (0.8/0.3/7)", base),
                ("+ L1 consensus", l1),
                ("+ L1 + L4 deep-rescue", l14),
                ("+ L1 + L4 + L5 (gt-tuned override) = s2hy7", l145)]:
    r1, r5, r10, mp = sc(x)
    print(f"{name:48s} R@1={r1:6.2f}  R@5={r5:6.2f}  R@10={r10:6.2f}  mAP={mp:6.2f}")
print(f"\nL5 fired on {fired} queries")
