"""Encode gallery (RGB) + query text with a frozen SigLIP and save sims_ext (1978x36773)
for ensemble fusion. Runs inside ssdc-stage1 (transformers 4.44.2 supports SiglipModel).

  python tools/encode_siglip.py --model models_ext/siglip-so400m \
     --gallery data/localeval_dedup/gallery.jsonl \
     --queries data/localeval_dedup/queries_competition.jsonl \
     --gt submissions/gt.json --name ext_siglip --gpu 0
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def _load_img(p):
    from PIL import Image
    try:
        return Image.open(p).convert('RGB')
    except Exception:
        return Image.new('RGB', (384, 384))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--gallery', required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--gt', default='submissions/gt.json')
    ap.add_argument('--name', default='ext_siglip')
    ap.add_argument('--out_root', default='submissions')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--fp16', action='store_true', help="load in fp16 (default fp32 reference)")
    ap.add_argument('--ckpt_dir', default=None, help="save gallery embeds as shards here (survive OOM)")
    ap.add_argument('--ckpt_every', type=int, default=1600)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=0)
    ap.add_argument('--gallery_only', action='store_true')
    a = ap.parse_args()
    if a.gpu < 0:                       # CPU finalize (merge cached shards + text + sims, no GPU)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'; dtype = torch.float32
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
        device = 'cuda'
        dtype = torch.float16 if a.fp16 else torch.float32

    from transformers import SiglipModel, AutoProcessor
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model = SiglipModel.from_pretrained(a.model, torch_dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(a.model)
    print(f"### loaded {a.model}", flush=True)

    gal = load_jsonl(a.gallery)
    names = np.array([os.path.basename(r['rgb']) for r in gal])
    paths = [r['rgb'] for r in gal]

    shards = {}
    if a.ckpt_dir:
        os.makedirs(a.ckpt_dir, exist_ok=True)
        for fn in os.listdir(a.ckpt_dir):
            if fn.startswith('gemb_') and fn.endswith('.pt'):
                shards[int(fn[5:-3])] = torch.load(os.path.join(a.ckpt_dir, fn), map_location='cpu')
        if shards:
            print(f"### resume: {len(shards)} shards", flush=True)

    def covered(idx):
        return any(st <= idx < st + t.shape[0] for st, t in shards.items())

    lo = a.start
    hi = a.end if a.end > 0 else len(paths)
    done = sum(t.shape[0] for t in shards.values())
    pool = ThreadPoolExecutor(max_workers=8)
    buf, buf_start = [], None
    with torch.no_grad():
        for s in range(lo, hi, a.bs):
            if covered(s):
                continue
            if buf_start is None:
                buf_start = s
            imgs = list(pool.map(_load_img, paths[s:min(s + a.bs, hi)]))
            px = proc(images=imgs, return_tensors='pt')['pixel_values'].to(device, dtype)
            f = model.get_image_features(pixel_values=px)
            buf.append(F.normalize(f.float(), dim=-1).cpu())
            done += len(imgs)
            if done % 3200 < a.bs:
                print(f"  img {done}/{len(paths)}", flush=True)
            if a.ckpt_dir and sum(t.shape[0] for t in buf) >= a.ckpt_every:
                sh = torch.cat(buf); shards[buf_start] = sh
                torch.save(sh, os.path.join(a.ckpt_dir, f'gemb_{buf_start}.pt'))
                buf, buf_start = [], None
    if a.ckpt_dir and buf:
        sh = torch.cat(buf); shards[buf_start] = sh
        torch.save(sh, os.path.join(a.ckpt_dir, f'gemb_{buf_start}.pt'))
    pool.shutdown()
    if a.gallery_only:
        cov = set()
        for st, t in shards.items():
            cov.update(range(st, min(st + t.shape[0], len(paths))))
        print(f"### gallery_only [{lo},{hi}) done; distinct covered {len(cov)}/{len(paths)}", flush=True)
        return
    if a.ckpt_dir:
        D = next(iter(shards.values())).shape[1]
        G = torch.zeros(len(paths), D); filled = torch.zeros(len(paths), dtype=torch.bool)
        for st in sorted(shards):
            t = shards[st]; e = min(st + t.shape[0], len(paths))
            G[st:e] = t[:e - st]; filled[st:e] = True
        miss = int((~filled).sum())
        if miss:
            raise SystemExit(f"### FATAL: {miss} gallery rows uncovered; rerun to fill")
        print(f"### merged {tuple(G.shape)} from {len(shards)} shards", flush=True)
    else:
        G = torch.cat(buf)
    print(f"### gallery encoded {tuple(G.shape)}", flush=True)

    recs = load_jsonl(a.queries)
    qidx = [r['query_index_comp'] for r in recs]
    caps = [r['caption'] for r in recs]

    TBS = 64
    tshards = {}
    if a.ckpt_dir:
        for fn in os.listdir(a.ckpt_dir):
            if fn.startswith('txt_') and fn.endswith('.pt'):
                tshards[int(fn[4:-3])] = torch.load(os.path.join(a.ckpt_dir, fn), map_location='cpu')
        if tshards:
            print(f"### text resume: {sum(t.shape[0] for t in tshards.values())} caps done", flush=True)
    def tcovered(idx):
        return any(st <= idx < st + t.shape[0] for st, t in tshards.items())
    with torch.no_grad():
        for s in range(0, len(caps), TBS):
            if tcovered(s):
                continue
            e = min(s + TBS, len(caps))
            t = proc(text=caps[s:e], padding='max_length', max_length=64,
                     truncation=True, return_tensors='pt').to(device)
            f = model.get_text_features(**t)
            emb = F.normalize(f.float(), dim=-1).cpu()
            if a.ckpt_dir:
                torch.save(emb, os.path.join(a.ckpt_dir, f'txt_{s}.pt'))
            tshards[s] = emb
            if s % 3200 < TBS:
                print(f"  txt {e}/{len(caps)}", flush=True)
    if a.ckpt_dir:
        D = next(iter(tshards.values())).shape[1]
        T = torch.zeros(len(caps), D)
        for st in sorted(tshards):
            t = tshards[st]; e = min(st + t.shape[0], len(caps)); T[st:e] = t[:e - st]
    else:
        T = torch.cat([tshards[st] for st in sorted(tshards)])
    sims = (T @ G.t()).numpy().astype(np.float32)
    print(f"### sims_ext {sims.shape}", flush=True)

    run = os.path.join(a.out_root, a.name, 'features')
    os.makedirs(run, exist_ok=True)
    np.save(os.path.join(run, 'sims_ext.npy'), sims)
    json.dump({'names': [str(x) for x in names], 'qidx': list(qidx), 'model': a.model},
              open(os.path.join(run, 'index.json'), 'w'), ensure_ascii=False)
    print(f"### saved -> {run}", flush=True)

    # standalone eval of this member
    gt = json.load(open(a.gt))
    col = {os.path.splitext(n)[0]: i for i, n in enumerate(names)}
    order = np.argsort(-sims, axis=1)
    pos = []
    for i, q in enumerate(qidx):
        c = col[gt[q]['gt_token']]
        pos.append(int(np.where(order[i] == c)[0][0]) + 1)
    pos = np.array(pos)
    print("### SigLIP-only:", {k: round(100 * (pos <= k).mean(), 2) for k in (1, 5, 10, 20, 50, 128)}, flush=True)


if __name__ == '__main__':
    main()
