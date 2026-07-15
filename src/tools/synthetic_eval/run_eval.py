"""ITC retrieval eval on the synthetic val split.

Reads val_gallery.jsonl + val_queries_{split}.jsonl from --eval-dir,
runs ITC-only retrieval (no ITM rerank), and prints R@1/R@5/R@10/mAP.
Appends a row to {eval-dir}/results.csv.

Run inside ssdc-eval container (cwd /workspace/SSDC):
  python tools/synthetic_eval/run_eval.py \
    --config configs/ssdc_openpose_ver2.yaml \
    --checkpoint checkpoint/cmp.pth \
    --eval-dir data/synthetic_eval \
    --image-root data/PAB/ \
    --split clean \
    --config-tag cmp_baseline_clean
"""
import os
import re
import csv
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure repo root is on sys.path when run from any cwd
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers copied verbatim from infer_submit.py
# ---------------------------------------------------------------------------

def pre_caption(c, mw):
    c = re.sub(r"([,.'!?\"()*#:;~])", '', c.lower()).replace('-', ' ').replace('/', ' ')
    c = re.sub(r"\s{2,}", ' ', c).rstrip('\n').strip(' ')
    w = c.split()
    return ' '.join(w[:mw]) if len(w) > mw else c


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


class GalleryDS(Dataset):
    def __init__(self, items, t, blank_pose=False):
        self.items, self.t, self.blank_pose = items, t, blank_pose

    def __len__(self):
        return len(self.items)

    def _load(self, p):
        from PIL import Image
        try:
            return self.t(Image.open(p).convert('RGB'))
        except Exception as e:
            print(f"  !! bad image {p}: {e} -> black", flush=True)
            return self.t(Image.new('RGB', (64, 64)))

    def __getitem__(self, i):
        from PIL import Image
        r = self.items[i]
        # blank-pose ablation: feed a black pose to measure the pose branch's contribution
        pose = self.t(Image.new('RGB', (64, 64))) if self.blank_pose else self._load(r["pose"])
        return self._load(r["rgb"]), pose


@torch.no_grad()
def encode_gallery(model, items, t, device, bs=32, nw=0, blank_pose=False, return_embeds=False):
    dl = DataLoader(GalleryDS(items, t, blank_pose=blank_pose), batch_size=bs, num_workers=nw, pin_memory=False)
    feats, embeds, done = [], [], 0
    for img, pose in dl:
        img, pose = img.to(device), pose.to(device)
        emb, _ = model.get_vision_embeds(img)
        if model.be_pose_conv:
            pose = model.pose_conv(pose)
        pemb, _ = model.get_vision_embeds(pose)
        emb = model.pose_block(emb, pemb)
        feats.append(F.normalize(model.get_image_feat(emb), dim=-1).cpu())
        if return_embeds:
            embeds.append(emb.cpu())          # [b,50,1024] full seq -> ITM cross-encoder
        done += img.size(0)
        if done % 3200 == 0:
            print(f"  gallery {done}/{len(items)}", flush=True)
    return torch.cat(feats), (torch.cat(embeds) if return_embeds else None)   # [G,D], [G,50,1024]|None


@torch.no_grad()
def encode_text_embeds(model, tok, caps, device, mw, bs=128):
    """Full text token embeds + attention masks (for ITM cross-encoder). Mirrors infer_submit."""
    embs, atts = [], []
    for s in range(0, len(caps), bs):
        b = [pre_caption(c, mw) for c in caps[s:s + bs]]
        tk = tok(b, padding='max_length', truncation=True, max_length=mw, return_tensors='pt').to(device)
        embs.append(model.get_text_embeds(tk.input_ids, tk.attention_mask).cpu())
        atts.append(tk.attention_mask.cpu())
    return torch.cat(embs), torch.cat(atts)


@torch.no_grad()
def itm_rerank(model, device, sims, gembeds, temb, tatt, top_k=128):
    """Full stage-1: re-score each query's top_k ITC candidates with the CMP ITM cross-encoder
    head (get_cross_embeds -> itm_head). Mirrors infer_submit.itm_rerank exactly."""
    nq, ng = sims.shape
    te, ta = temb.to(device), tatt.to(device)
    out = torch.full((nq, ng), -1e4)
    st = torch.from_numpy(sims)
    for i in range(nq):
        _, idx = st[i].topk(k=min(top_k, ng))
        enc = gembeds[idx].to(device)
        att = torch.ones(enc.size()[:-1], dtype=torch.long, device=device)
        o = model.get_cross_embeds(enc, att,
                                   text_embeds=te[i].unsqueeze(0).repeat(len(idx), 1, 1),
                                   text_atts=ta[i].unsqueeze(0).repeat(len(idx), 1))[:, 0, :]
        out[i, idx] = model.itm_head(o)[:, 1].cpu()
        if (i + 1) % 2000 == 0:
            print(f"  itm {i+1}/{nq}", flush=True)
    return out.numpy()


@torch.no_grad()
def encode_text_feats(model, tok, caps, device, mw, bs=256):
    out = []
    for s in range(0, len(caps), bs):
        b = [pre_caption(c, mw) for c in caps[s:s + bs]]
        t = tok(b, padding='max_length', truncation=True, max_length=mw, return_tensors='pt').to(device)
        emb = model.get_text_embeds(t.input_ids, t.attention_mask)
        out.append(F.normalize(model.get_text_feat(emb), dim=-1).cpu())
    return torch.cat(out)   # [Q, D]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Synthetic val ITC retrieval eval")
    ap.add_argument('--config', default='configs/ssdc_openpose_ver2.yaml')
    ap.add_argument('--checkpoint', default='checkpoint/cmp.pth')
    ap.add_argument('--adapter', default='',
                    help="path to a text-tower LoRA adapter dir; injected into BERT (consistency "
                         "fine-tune). Vision frozen -> gallery feats unchanged. Mirrors infer_submit.")
    ap.add_argument('--adapter-joint', action='store_true',
                    help='load the adapter as a whole-model joint LoRA (Swin+BERT), not text-only')
    ap.add_argument('--stage1-adapter', default='',
                    help='frozen Stage-1 joint LoRA (vision_only) merged in before the Stage-2 ITM adapter')
    ap.add_argument('--stage2-adapter', default='',
                    help='Stage-2 ITM LoRA (BERT 6-11 + itm_head); requires --stage1-adapter + --itm')
    ap.add_argument('--eval-dir', default='data/synthetic_eval')
    ap.add_argument('--image-root', default='data/PAB/')
    ap.add_argument('--split', choices=['clean', 'noisy'], default='clean')
    ap.add_argument('--config-tag', default='cmp_baseline')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8,
                    help="DataLoader workers (ssdc-eval has 32g shm so >0 is safe; "
                         "infer_submit used 0 because its container had only 64M /dev/shm)")
    ap.add_argument('--degrade-gallery', action='store_true',
                    help="apply generic RandomDownscale(64-160px)+JPEG to gallery images on the fly "
                         "(forces nw=0). Prefer the frozen path: dump_degraded_gallery.py + "
                         "--gallery-jsonl/--rgb-root (faster, workers>0, byte-stable). Don't combine.")
    ap.add_argument('--degrade-seed', type=int, default=20260622)
    ap.add_argument('--gallery-jsonl', default=None,
                    help="override gallery manifest (default {eval-dir}/val_gallery.jsonl); "
                         "point at val_gallery_degraded.jsonl to read a frozen pre-degraded gallery")
    ap.add_argument('--rgb-root', default=None,
                    help="prefix joined to a manifest 'rgb' field (frozen degraded gallery); "
                         "pose still resolves from --image-root + original 'image'")
    ap.add_argument('--dump-scores', default=None,
                    help='dir: dump per-query top-K candidate ids + ITC (and ITM) scores '
                         'for the learned-fusion feature table')
    ap.add_argument('--blank-pose', action='store_true',
                    help="feed a black pose to every gallery item -> ablates the pose branch "
                         "(R@1 vs real-pose run = pose's contribution; B0 of the pose plan)")
    ap.add_argument('--itm', action='store_true',
                    help="full CMP stage-1: ITM cross-encoder rerank of the top_k ITC candidates "
                         "(get_cross_embeds + itm_head). Off = ITC only. Excludes the Qwen agent (stage-2).")
    ap.add_argument('--top_k', type=int, default=128, help="ITC candidates per query reranked by ITM")
    a = ap.parse_args()

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from transformers import BertTokenizer
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"### device: {device}", flush=True)

    # --- Config + Model ---
    config = yaml.load(open(a.config))
    print(f"### config: {a.config}  checkpoint: {a.checkpoint}", flush=True)

    from models.model_search import Search
    model = Search(config=config)
    model.load_pretrained(a.checkpoint)
    if a.stage2_adapter:                                 # Stage-2 ITM on frozen merged Stage-1
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        from models.lora_text import apply_joint_lora, apply_itm_lora
        model = apply_joint_lora(model, r=config.get('stage1_lora_r', 16), alpha=config.get('stage1_lora_alpha', 32),
                                 dropout=config.get('lora_dropout', 0.05),
                                 proj_heads=config.get('stage1_proj_heads', 'vision_only'))
        set_peft_model_state_dict(model, load_file(os.path.join(a.stage1_adapter, 'adapter_model.safetensors')))
        model = model.merge_and_unload()
        model = apply_itm_lora(model, r=config.get('lora_r', 16), alpha=config.get('lora_alpha', 32),
                               dropout=config.get('lora_dropout', 0.05))
        set_peft_model_state_dict(model, load_file(os.path.join(a.stage2_adapter, 'adapter_model.safetensors')))
        print(f'### Stage-1 {a.stage1_adapter} merged + Stage-2 ITM {a.stage2_adapter}', flush=True)
    elif a.adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(os.path.join(a.adapter, 'adapter_model.safetensors'))
        if a.adapter_joint:
            from models.lora_text import apply_joint_lora
            model = apply_joint_lora(model, r=config.get('lora_r',16), alpha=config.get('lora_alpha',32),
                                     dropout=config.get('lora_dropout',0.05),
                                     proj_heads=config.get('lora_proj_heads','none'))
            set_peft_model_state_dict(model, sd)
        else:                                            # legacy text-only adapter
            from models.lora_text import apply_text_lora
            apply_text_lora(model, r=config.get('lora_r',16), alpha=config.get('lora_alpha',32),
                            dropout=config.get('lora_dropout',0.05))
            set_peft_model_state_dict(model.text_encoder, sd)
        print(f'### injected adapter (joint={a.adapter_joint}): {a.adapter}', flush=True)
    model = model.to(device).eval()

    tok = BertTokenizer.from_pretrained(config['text_encoder'])
    mw = config.get('max_words', 56)

    transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    # --- Gallery ---
    gal_path = a.gallery_jsonl or os.path.join(a.eval_dir, 'val_gallery.jsonl')
    gal = load_jsonl(gal_path)
    print(f"### gallery: {len(gal)} items from {gal_path}", flush=True)

    image_root = a.image_root
    items = []
    for r in gal:
        if r.get("rgb"):   # frozen pre-degraded gallery: rgb already written to disk
            rgb_path = os.path.join(a.rgb_root, r["rgb"]) if a.rgb_root else r["rgb"]
        else:
            rgb_path = os.path.join(image_root, r["image"])
        if r.get("pose"):   # manifest-provided pose (e.g. image_id-keyed val-hard pose), joined to --rgb-root
            pose_path = os.path.join(a.rgb_root, r["pose"]) if a.rgb_root else r["pose"]
        else:
            pose_path = os.path.join(image_root, "pose", r["image"])   # default tree, pose never degraded
        items.append({"rgb": rgb_path, "pose": pose_path})

    gal_t, gal_nw = transform, a.workers
    if a.degrade_gallery:
        from datasets.build import RandomDownscale, RandomJPEG
        import random as _r
        _r.seed(a.degrade_seed)  # + nw=0 below -> deterministic, reproducible degraded gallery
        gal_t = transforms.Compose([
            RandomDownscale(p=1.0, smin=64, smax=160, size=(config['h'], config['w'])),
            RandomJPEG(p=1.0, qmin=30, qmax=85),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        gal_nw = 0
        print("### gallery DEGRADED (downscale 64-160px + JPEG q30-85, seeded)", flush=True)
    print(f"### encoding gallery (bs={a.bs}, workers={gal_nw}, degrade={a.degrade_gallery})...", flush=True)
    gfeat, gembeds = encode_gallery(model, items, gal_t, device, bs=a.bs, nw=gal_nw,
                                    blank_pose=a.blank_pose, return_embeds=a.itm)  # [G,D], embeds|None
    print(f"### gallery encoded: {gfeat.shape}", flush=True)

    # Build image_id -> gallery row index map
    gal_id2idx = {r["image_id"]: i for i, r in enumerate(gal)}

    # --- Queries ---
    qry_path = os.path.join(a.eval_dir, f'val_queries_{a.split}.jsonl')
    queries = load_jsonl(qry_path)
    print(f"### queries: {len(queries)} items from {qry_path}", flush=True)

    caps = [r["caption"] for r in queries]

    # Validate all query image_ids exist in gallery
    missing = [r["image_id"] for r in queries if r["image_id"] not in gal_id2idx]
    if missing:
        print(f"  !! {len(missing)} query image_ids missing from gallery. First: {missing[:5]}", flush=True)
        sys.exit(1)

    gt_idx = np.array([gal_id2idx[r["image_id"]] for r in queries], dtype=np.int64)

    print(f"### encoding text...", flush=True)
    tfeat = encode_text_feats(model, tok, caps, device, mw)  # [Q, D]
    print(f"### text encoded: {tfeat.shape}", flush=True)

    # --- Compute similarities ---
    print("### computing ITC similarities...", flush=True)
    sims = (tfeat @ gfeat.t()).numpy()   # [Q, G]
    print(f"### sims shape: {sims.shape}", flush=True)
    itc_sims = sims if a.dump_scores else None   # keep pre-ITM ITC matrix for the dump

    from tools.synthetic_eval.metrics import _ranks
    if a.itm:                            # full CMP stage-1: ITM cross-encoder rerank of top_k ITC
        itc_rk = _ranks(sims, gt_idx)
        print(f"### ITC-only: R@1 {float((itc_rk < 1).mean() * 100):.2f} "
              f"mAP {float(np.mean(1.0 / (itc_rk + 1)) * 100):.2f}  -> ITM rerank top_k={a.top_k}...", flush=True)
        temb, tatt = encode_text_embeds(model, tok, caps, device, mw)
        sims = itm_rerank(model, device, sims, gembeds, temb, tatt, top_k=a.top_k)
        print("### ITM rerank done (metrics below = full stage-1 ITC+ITM)", flush=True)

    if a.dump_scores:
        # per-query top-K candidates by ITC, with both ITC and (if computed) ITM scores —
        # feature source for the learned (LambdaMART) fusion; ids not row indices
        os.makedirs(a.dump_scores, exist_ok=True)
        K = min(a.top_k, itc_sims.shape[1])
        it = torch.from_numpy(itc_sims)
        topv, topi = it.topk(K, dim=1)
        gal_ids = np.array([r["image_id"] for r in gal])
        out = {"K": K, "queries": [], "gallery_ids": gal_ids[topi.numpy()].tolist(),
               "itc": topv.numpy().astype(np.float32).tolist()}
        if a.itm:
            itm_v = torch.from_numpy(sims).gather(1, topi).numpy().astype(np.float32)
            out["itm"] = itm_v.tolist()
        out["queries"] = [{"i": i, "image_id": r["image_id"], "level": r.get("level"),
                           "style": r.get("style")} for i, r in enumerate(queries)]
        import json as _json
        with open(os.path.join(a.dump_scores, 'cmp_scores.json'), 'w') as f:
            _json.dump(out, f)
        print(f"### dumped top-{K} CMP scores -> {a.dump_scores}/cmp_scores.json", flush=True)

    # --- Metrics: overall + per difficulty-level bin. Rank once, slice by query 'level' tag. ---
    rk = _ranks(sims, gt_idx)

    def _metrics(r):
        return {'R@1': float((r < 1).mean()*100), 'R@5': float((r < 5).mean()*100),
                'R@10': float((r < 10).mean()*100), 'recall@128': float((r < 128).mean()*100),
                'mAP': float(np.mean(1.0/(r+1))*100)}

    overall = _metrics(rk)
    levels = np.array([q.get('level', 'all') for q in queries])
    uniq = set(levels.tolist())
    bin_order = [l for l in ('easy', 'medium', 'hard') if l in uniq] + \
                sorted(u for u in uniq if u not in ('easy', 'medium', 'hard'))
    bins = {lvl: _metrics(rk[levels == lvl]) for lvl in bin_order}

    print("\n" + "=" * 60, flush=True)
    print(f"  config-tag : {a.config_tag}   split: {a.split}", flush=True)
    print(f"  gallery    : {len(gal)}  queries: {len(queries)}", flush=True)
    print(f"  OVERALL    : R@1 {overall['R@1']:.2f}  R@10 {overall['R@10']:.2f}  "
          f"recall@128 {overall['recall@128']:.2f}  mAP {overall['mAP']:.2f}", flush=True)
    if len(bins) > 1:
        for lvl in bin_order:
            n = int((levels == lvl).sum())
            print(f"    [{lvl:7s}] R@1 {bins[lvl]['R@1']:.2f}  mAP {bins[lvl]['mAP']:.2f}  (n={n})", flush=True)
    print("=" * 60 + "\n", flush=True)

    # --- results.csv (overall + per-bin R@1). Rewrite whole file each run so the header stays
    #     consistent when columns evolve (old rows get padded with empty new columns). ---
    csv_path = os.path.join(a.eval_dir, 'results.csv')
    fields = ['config_tag', 'split', 'R@1', 'R@5', 'R@10', 'recall@128', 'mAP', 'R@1_easy', 'R@1_medium', 'R@1_hard']
    row = {'config_tag': a.config_tag, 'split': a.split,
           'R@1': f"{overall['R@1']:.2f}", 'R@5': f"{overall['R@5']:.2f}",
           'R@10': f"{overall['R@10']:.2f}", 'recall@128': f"{overall['recall@128']:.2f}",
           'mAP': f"{overall['mAP']:.2f}",
           'R@1_easy': f"{bins['easy']['R@1']:.2f}" if 'easy' in bins else '',
           'R@1_medium': f"{bins['medium']['R@1']:.2f}" if 'medium' in bins else '',
           'R@1_hard': f"{bins['hard']['R@1']:.2f}" if 'hard' in bins else ''}
    prev = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            prev = list(csv.DictReader(f))
    prev.append(row)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in prev:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f"### results -> {csv_path}", flush=True)


if __name__ == '__main__':
    main()
