"""Encode gallery (RGB) + query text with a frozen OpenCLIP ViT-H/14 (laion2b) and save
sims_ext (1978x36773) for ensemble fusion. Loaded via HF transformers CLIPModel (no open_clip
dep needed); runs inside ssdc-stage1 (transformers 4.44.2 supports CLIPModel). ViT-H text
context = 77 tokens (long queries get truncated -- expected).

  HF_HOME=/workspace/SSDC/models_ext/hf_cache CUDA_VISIBLE_DEVICES=3 \
  python tools/encode_openclip.py --model laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
     --gallery data/localeval_dedup/gallery.jsonl \
     --queries data/localeval_dedup/queries_competition.jsonl \
     --gt submissions/gt.json --name ext_openclip_vith --fp16
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
        return Image.new('RGB', (224, 224))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    ap.add_argument('--gallery', required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--gt', default='submissions/gt.json')
    ap.add_argument('--name', default='ext_openclip_vith')
    ap.add_argument('--out_root', default='submissions')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max_text_len', type=int, default=77, help="CLIP context length")
    ap.add_argument('--fp16', action='store_true', help="load in fp16 (default fp32 reference)")
    ap.add_argument('--ckpt_dir', default=None,
                    help="if set, save gallery embeds as shards here (survive OOM; resume on restart)")
    ap.add_argument('--ckpt_every', type=int, default=1600, help="checkpoint every N images")
    ap.add_argument('--start', type=int, default=0, help="gallery slice start (data-parallel shard)")
    ap.add_argument('--end', type=int, default=0, help="gallery slice end (0=to the end)")
    ap.add_argument('--gallery_only', action='store_true',
                    help="encode only the gallery slice + checkpoint, then exit (for sharded runs)")
    a = ap.parse_args()
    # gpu < 0 -> CPU finalize (merge cached shards + text encode + sims with no GPU); used to
    # sidestep contention when the gallery is already checkpointed and only the tiny text pass
    # remains. fp16 not supported on CPU, so force fp32.
    if a.gpu < 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'; dtype = torch.float32
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
        device = 'cuda'
        dtype = torch.float16 if a.fp16 else torch.float32

    from transformers import CLIPModel, AutoProcessor
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model = CLIPModel.from_pretrained(a.model, torch_dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(a.model)
    print(f"### loaded {a.model}  (dtype={dtype})", flush=True)

    gal = load_jsonl(a.gallery)
    names = np.array([os.path.basename(r['rgb']) for r in gal])
    paths = [r['rgb'] for r in gal]

    # resume: each shard is gemb_{start}.pt covering images [start, start+len). Load all
    # existing shards, then skip to the first uncovered image so an OOM mid-run costs at
    # most one ckpt_every window instead of the whole gallery.
    shards = {}
    if a.ckpt_dir:
        os.makedirs(a.ckpt_dir, exist_ok=True)
        for fn in os.listdir(a.ckpt_dir):
            if fn.startswith('gemb_') and fn.endswith('.pt'):
                st = int(fn[5:-3])
                shards[st] = torch.load(os.path.join(a.ckpt_dir, fn), map_location='cpu')
        if shards:
            print(f"### resume: {len(shards)} shards, {sum(t.shape[0] for t in shards.values())} imgs done", flush=True)

    def covered(idx):
        for st, t in shards.items():
            if st <= idx < st + t.shape[0]:
                return True
        return False

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
            cur = sum(t.shape[0] for t in buf)
            if a.ckpt_dir and cur >= a.ckpt_every:
                shard = torch.cat(buf)
                shards[buf_start] = shard
                torch.save(shard, os.path.join(a.ckpt_dir, f'gemb_{buf_start}.pt'))
                buf, buf_start = [], None
    if a.ckpt_dir and buf:
        shard = torch.cat(buf)
        shards[buf_start] = shard
        torch.save(shard, os.path.join(a.ckpt_dir, f'gemb_{buf_start}.pt'))
    pool.shutdown()
    if a.gallery_only:
        # count DISTINCT covered indices (shards from different runs may overlap)
        cov = set()
        for st, t in shards.items():
            cov.update(range(st, min(st + t.shape[0], len(paths))))
        print(f"### gallery_only shard [{lo},{hi}) done; distinct covered {len(cov)}/{len(paths)}", flush=True)
        return
    if a.ckpt_dir:
        # place each shard at its absolute [st:st+n) slice so overlapping/misaligned shards
        # (e.g. bs=8 sharded run + bs=32 single-GPU run) collapse to exactly N rows in order
        D = next(iter(shards.values())).shape[1]
        G = torch.zeros(len(paths), D)
        filled = torch.zeros(len(paths), dtype=torch.bool)
        for st in sorted(shards):
            t = shards[st]; e = min(st + t.shape[0], len(paths))
            G[st:e] = t[:e - st]; filled[st:e] = True
        miss = int((~filled).sum())
        if miss:
            raise SystemExit(f"### FATAL: {miss} gallery rows uncovered by shards; rerun to fill")
        print(f"### merged {tuple(G.shape)} from {len(shards)} shards (all {len(paths)} rows filled)", flush=True)
    else:
        G = torch.cat(buf)
    print(f"### gallery encoded {tuple(G.shape)}", flush=True)

    recs = load_jsonl(a.queries)
    qidx = [r['query_index_comp'] for r in recs]
    caps = [r['caption'] for r in recs]

    # text-embedding checkpoint (same idea as gallery): under brutal GPU contention the text
    # pass can only run in short windows, so save shards and resume — progress survives OOM.
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
            t = proc(text=caps[s:e], padding='max_length', max_length=a.max_text_len,
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
    print("### ViT-H-only:", {k: round(100 * (pos <= k).mean(), 2) for k in (1, 5, 10, 20, 50, 128)}, flush=True)


if __name__ == '__main__':
    main()
