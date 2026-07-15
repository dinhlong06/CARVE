"""THE single submission scorer (offline, no model). Replaces run_localeval metrics,
selfscore_answer, eval_tta. Importable (infer_submit calls score()) and runnable as CLI.

  python3 eval_submission.py submissions/<name>      # reads <name>/submission.json + ../gt.json

submission: {query_index: [gallery filename, ...]}  (filenames may carry .jpg)
gt:         {query_index: {"gt_token": <gallery stem>, ...}}
Each query has exactly one relevant gallery image -> mAP == MRR (capped at the stored list).
"""
import os
import sys
import json


def _stem(s):
    return os.path.splitext(s)[0]


def score(submission, gt):
    R = {1: 0, 5: 0, 10: 0}
    ap = 0.0
    n = 0
    for qi, g in gt.items():
        ranked = [_stem(x) for x in submission.get(qi, [])]
        tok = g["gt_token"]
        rank = ranked.index(tok) + 1 if tok in ranked else 10 ** 9
        for k in R:
            R[k] += rank <= k
        ap += 1.0 / rank if rank <= 10 else 0.0
        n += 1
    return {"R@1": round(100 * R[1] / n, 2), "R@5": round(100 * R[5] / n, 2),
            "R@10": round(100 * R[10] / n, 2), "mAP": round(100 * ap / n, 2), "n": n}


def score_by_change(submission, gt):
    """Per-perturbation-type breakdown of R@1/R@5/R@10/mAP, using gt[qi]['change'].
    Returns {change_type: {R@1,R@5,R@10,mAP,n}} sorted by descending n. Queries whose
    gt entry lacks a 'change' field are bucketed under '<none>'."""
    buckets = {}
    for qi, g in gt.items():
        ch = g.get("change", "<none>")
        ranked = [_stem(x) for x in submission.get(qi, [])]
        tok = g["gt_token"]
        rank = ranked.index(tok) + 1 if tok in ranked else 10 ** 9
        b = buckets.setdefault(ch, {1: 0, 5: 0, 10: 0, "ap": 0.0, "n": 0})
        for k in (1, 5, 10):
            b[k] += rank <= k
        b["ap"] += 1.0 / rank if rank <= 10 else 0.0
        b["n"] += 1
    out = {}
    for ch, b in sorted(buckets.items(), key=lambda kv: -kv[1]["n"]):
        n = b["n"]
        out[ch] = {"R@1": round(100 * b[1] / n, 2), "R@5": round(100 * b[5] / n, 2),
                   "R@10": round(100 * b[10] / n, 2), "mAP": round(100 * b["ap"] / n, 2), "n": n}
    return out


def coverage(scores, gt, ks=(1, 5, 10, 20, 30, 50)):
    """Recall@k over the stage-1 candidate POOL (scores.json) -> the stage-2 ceiling.
    scores: {query_index: [[gallery_filename, stage1_score], ...]}.
    R@k = % queries whose gt_token sits within the top-k of the pool. R@<pool> = is the
    pool wide enough to contain the answer at all (100 = fully covered)."""
    pool = max((len(v) for v in scores.values()), default=0)
    ks = sorted({k for k in ks if k <= pool} | ({pool} if pool else set()))
    hit = {k: 0 for k in ks}
    n = 0
    for qi, g in gt.items():
        lst = [_stem(x[0]) for x in scores.get(qi, [])]
        tok = g["gt_token"]
        rank = lst.index(tok) + 1 if tok in lst else 10 ** 9
        for k in ks:
            hit[k] += rank <= k
        n += 1
    out = {"pool": pool, "n": n}
    for k in ks:
        out[f"R@{k}"] = round(100 * hit[k] / n, 2)
    return out


def evaluate_run(run):
    """Score a run dir against ../gt.json -> {metrics?, coverage?} (whichever files exist)."""
    run = run.rstrip("/")
    gt = json.load(open(os.path.join(os.path.dirname(run), "gt.json")))
    res = {}
    sub_p = os.path.join(run, "submission.json")
    if os.path.exists(sub_p):
        sub = json.load(open(sub_p))
        res["metrics"] = score(sub, gt)
        res["by_change"] = score_by_change(sub, gt)
    sc_p = os.path.join(run, "scores.json")
    if os.path.exists(sc_p):
        res["coverage"] = coverage(json.load(open(sc_p)), gt)
    return res


def main():
    run = sys.argv[1].rstrip("/")
    res = evaluate_run(run)
    if "metrics" in res:
        print("submission metrics:", json.dumps(res["metrics"]))
    if "by_change" in res:
        print("by change type (sorted by n):")
        print(f"  {'change':<10} {'n':>4} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'mAP':>7}")
        for ch, m in res["by_change"].items():
            print(f"  {ch:<10} {m['n']:>4} {m['R@1']:>7} {m['R@5']:>7} {m['R@10']:>7} {m['mAP']:>7}")
    if "coverage" in res:
        print("pool coverage     :", json.dumps(res["coverage"]))
    # persist back into meta.json so it can be reviewed later
    mp = os.path.join(run, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {"name": os.path.basename(run)}
    meta.update(res)
    json.dump(meta, open(mp, "w"), ensure_ascii=False, indent=2)
    print(f"updated {mp}")


if __name__ == "__main__":
    main()
