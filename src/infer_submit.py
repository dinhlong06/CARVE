"""Unified inference -> submission run folder (+ auto-eval). Replaces run_localeval,
gen_submission, inference_testset. Run inside ssdc-stage1 docker, cwd /workspace/SSDC.

  python infer_submit.py --name openpose_itm \
    --config configs/ssdc_openpose_ver2.yaml --checkpoint checkpoint/cmp.pth \
    --gallery data/localeval_dedup/gallery.jsonl \
    --queries data/localeval_dedup/queries_competition.jsonl \
    --gt submissions/gt.json --top_n 10 --top_k 128 --gpu 0

Writes submissions/<name>/{submission.json, answer.txt, meta.json}.
"""
import os
import re
import json
import argparse
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# This container ships a 64M /dev/shm -> DataLoader workers Bus-error with the default
# (shared-memory) tensor sharing. file_system sharing passes worker tensors via temp files
# instead, so --num_workers > 0 is safe here (mirrors train_joint_lora.py).
torch.multiprocessing.set_sharing_strategy("file_system")

# Heavy deps (PIL/torchvision/transformers/ruamel) are imported lazily inside the functions
# that need them, so the pure helper rank_to_outputs stays importable without them.


def pre_caption(c, mw):
    c = re.sub(r"([,.'!?\"()*#:;~])", '', c.lower()).replace('-', ' ').replace('/', ' ')
    c = re.sub(r"\s{2,}", ' ', c).rstrip('\n').strip(' ')
    w = c.split()
    return ' '.join(w[:mw]) if len(w) > mw else c


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def rank_to_outputs(sims, names, qidx, top_n):
    """Pure: sims [nq x ng], names [ng] np.array, qidx list -> (submission dict, answer lines)."""
    submission, answer = {}, []
    for i in range(sims.shape[0]):
        order = np.argsort(-sims[i])[:top_n]
        ranked = [str(x) for x in names[order]]
        submission[qidx[i]] = ranked
        answer.append(' '.join(os.path.splitext(x)[0] for x in ranked))
    return submission, answer


def rank_to_scores(sims, names, qidx, pool_k):
    """Pure: top pool_k candidates per query WITH stage-1 scores, for stage-2 reranking.
    Returns {query_index: [[gallery_filename, stage1_score], ...]} (descending)."""
    out = {}
    for i in range(sims.shape[0]):
        order = np.argsort(-sims[i])[:pool_k]
        out[qidx[i]] = [[str(names[j]), float(sims[i][j])] for j in order]
    return out


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
        r = self.items[i]
        from PIL import Image
        pose = self.t(Image.new('RGB', (64, 64))) if self.blank_pose else self._load(r["pose"])
        return self._load(r["rgb"]), pose


@torch.no_grad()
def encode_gallery(model, items, t, device, bs=32, nw=0, keep_embeds=True, blank_pose=False, no_pose_block=False):  # nw>0 ok with file_system sharing (set at import)
    # keep_embeds=False -> ITC-only: skip the [G,50,1024] token embeds (no ITM), much lighter RAM/disk.
    # no_pose_block=True -> TRUE no-pose ablation: skip the pose_block fusion entirely (use raw image emb).
    dl = DataLoader(GalleryDS(items, t, blank_pose=blank_pose or no_pose_block), batch_size=bs, num_workers=nw, pin_memory=True)
    feats, embeds, done = [], [], 0
    for img, pose in dl:
        img, pose = img.to(device), pose.to(device)
        emb, _ = model.get_vision_embeds(img)
        if not no_pose_block:
            if model.be_pose_conv:
                pose = model.pose_conv(pose)
            pemb, _ = model.get_vision_embeds(pose)
            emb = model.pose_block(emb, pemb)
        feats.append(F.normalize(model.get_image_feat(emb), dim=-1).cpu())
        if keep_embeds:
            embeds.append(emb.cpu())
        done += img.size(0)
        if done % 3200 == 0:
            print(f"  gallery {done}/{len(items)}", flush=True)
    return torch.cat(feats), (torch.cat(embeds) if keep_embeds else None)


@torch.no_grad()
def encode_text_feats(model, tok, caps, device, mw, bs=256):
    out = []
    for s in range(0, len(caps), bs):
        b = [pre_caption(c, mw) for c in caps[s:s + bs]]
        t = tok(b, padding='max_length', truncation=True, max_length=mw, return_tensors='pt').to(device)
        emb = model.get_text_embeds(t.input_ids, t.attention_mask)
        out.append(F.normalize(model.get_text_feat(emb), dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def encode_text_embeds(model, tok, caps, device, mw, bs=128):
    embs, atts = [], []
    for s in range(0, len(caps), bs):
        b = [pre_caption(c, mw) for c in caps[s:s + bs]]
        t = tok(b, padding='max_length', truncation=True, max_length=mw, return_tensors='pt').to(device)
        embs.append(model.get_text_embeds(t.input_ids, t.attention_mask).cpu())
        atts.append(t.attention_mask.cpu())
    return torch.cat(embs), torch.cat(atts)


@torch.no_grad()
def itm_rerank(model, device, sims, gembeds, temb, tatt, top_k=128):
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
        if (i + 1) % 500 == 0:
            print(f"  itm {i+1}/{nq}", flush=True)
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', default='checkpoint/cmp.pth')
    ap.add_argument('--gallery', required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--gt', default='submissions/gt.json')
    ap.add_argument('--out_root', default='submissions')
    ap.add_argument('--top_n', type=int, default=10, help="candidates in submission.json / answer.txt")
    ap.add_argument('--pool_k', type=int, default=50,
                    help="size of stage-1 candidate pool saved to scores.json for stage-2 "
                         "(default 50; 0 = use top_n; recommend <=top_k=128 where ITM scores are meaningful)")
    ap.add_argument('--top_k', type=int, default=128)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bs', type=int, default=32, help='gallery encode batch size (lower on crowded GPUs)')
    ap.add_argument('--num_workers', type=int, default=0,
                    help="DataLoader workers for the gallery encode (file_system sharing -> safe on 64M shm)")
    ap.add_argument('--itc_only', action='store_true',
                    help="rank by ITC contrastive sims only, skip the ITM cross-encoder rerank")
    ap.add_argument('--blank_pose', action='store_true',
                    help="feed a black pose to the gallery -> ablates the pose CONTENT (pose_block still runs)")
    ap.add_argument('--no_pose_block', action='store_true',
                    help="skip the pose_block fusion entirely -> TRUE no-pose ablation (measures the block itself)")
    ap.add_argument('--clean', action='store_true',
                    help="retrieve with demo_caption (de-paraphrased clean) instead of caption")
    ap.add_argument('--adapter', default='',
                    help="path to a LoRA adapter dir; injected into the text encoder (consistency fine-tune)")
    ap.add_argument('--joint_adapter', default='',
                    help="path to a JOINT (vision+text) LoRA adapter dir (train_joint_lora.py). Patches the "
                         "vision tower too, so gallery feats change -> --gallery_feats cache is NOT reusable.")
    ap.add_argument('--gallery_feats', default='',
                    help="dir with cached gallery_feats.pt+gallery_embeds.pt+index.json; text-only LoRA leaves "
                         "the vision tower frozen so gallery feats are identical -> skip the 36773-image encode")
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    device = 'cuda'

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from transformers import BertTokenizer
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')

    config = yaml.load(open(a.config))
    from models.model_search import Search
    model = Search(config=config)
    model.load_pretrained(a.checkpoint)
    if a.adapter:                                    # inject text-tower LoRA consistency adapter
        from models.lora_text import apply_text_lora
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        apply_text_lora(model, r=config.get('lora_r', 16), alpha=config.get('lora_alpha', 32),
                        dropout=config.get('lora_dropout', 0.05))
        set_peft_model_state_dict(model.text_encoder,
                                  load_file(os.path.join(a.adapter, 'adapter_model.safetensors')))
        print('### injected LoRA adapter:', a.adapter, flush=True)
    if a.joint_adapter:                              # inject JOINT vision+text LoRA over the whole Search model
        assert not a.gallery_feats, "joint adapter changes the vision tower -> cached gallery_feats are stale"
        from models.lora_text import apply_joint_lora
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(os.path.join(a.joint_adapter, 'adapter_model.safetensors'))
        proj = 'vision_only' if any('vision_proj' in k for k in sd) else 'none'
        if any('text_proj' in k for k in sd):
            proj = 'both'
        acfg = json.load(open(os.path.join(a.joint_adapter, 'adapter_config.json')))  # r/alpha differ per run
        model = apply_joint_lora(model, r=acfg['r'], alpha=acfg['lora_alpha'],
                                 dropout=acfg['lora_dropout'], proj_heads=proj)
        miss = set_peft_model_state_dict(model, sd)
        print(f'### injected JOINT LoRA adapter: {a.joint_adapter} (proj_heads={proj}); load={miss}', flush=True)
    model = model.to(device).eval()
    tok = BertTokenizer.from_pretrained(config['text_encoder'])
    mw = config.get('max_words', 56)
    transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    gal = load_jsonl(a.gallery)
    names = np.array([os.path.basename(r["rgb"]) for r in gal])
    if a.gallery_feats:                          # reuse cached FROZEN gallery feats (text-only LoRA leaves vision unchanged)
        import torch as _t
        cidx = json.load(open(os.path.join(a.gallery_feats, "index.json")))
        assert cidx["names"] == [str(x) for x in names], "cached gallery order != gallery.jsonl order"
        gfeat = _t.load(os.path.join(a.gallery_feats, "gallery_feats.pt")).float()    # [G,2048] L2-normalized
        gembeds = _t.load(os.path.join(a.gallery_feats, "gallery_embeds.pt"))          # [G,50,1024]
        print(f"### gallery {len(gal)} | REUSED cached feats from {a.gallery_feats} (no encode)", flush=True)
    else:
        print(f"### gallery {len(gal)} | encoding (RGB+pose), num_workers={a.num_workers}...", flush=True)
        gfeat, gembeds = encode_gallery(model, gal, transform, device, bs=a.bs, nw=a.num_workers, blank_pose=a.blank_pose, no_pose_block=a.no_pose_block)
        print("### gallery encoded.", flush=True)

    recs = load_jsonl(a.queries)
    qidx = [r["query_index_comp"] for r in recs]
    cap_field = 'demo_caption' if a.clean else 'caption'
    if a.clean and 'demo_caption' not in recs[0]:
        print("  !! --clean set but 'demo_caption' missing -> falling back to 'caption'", flush=True)
        cap_field = 'caption'
    print(f"### using caption field: {cap_field}", flush=True)
    caps = [r[cap_field] for r in recs]
    tfeat = encode_text_feats(model, tok, caps, device, mw)
    sims_itc = (tfeat @ gfeat.t()).numpy()
    if a.itc_only:
        print("### ITC-only ranking (no ITM rerank)", flush=True)
        sims = sims_itc
    else:
        temb, tatt = encode_text_embeds(model, tok, caps, device, mw)
        sims = itm_rerank(model, device, sims_itc, gembeds, temb, tatt, top_k=a.top_k)

    submission, answer = rank_to_outputs(sims, names, qidx, a.top_n)
    run = os.path.join(a.out_root, a.name)
    os.makedirs(run, exist_ok=True)
    json.dump(submission, open(os.path.join(run, 'submission.json'), 'w'), ensure_ascii=False, indent=2)
    open(os.path.join(run, 'answer.txt'), 'w').write('\n'.join(answer) + '\n')

    # stage-1 candidate pool + scores (hook for stage-2 LLM rerank / lambda-blend)
    pool_k = a.pool_k if a.pool_k > 0 else a.top_n
    scores = rank_to_scores(sims, names, qidx, pool_k)
    json.dump(scores, open(os.path.join(run, 'scores.json'), 'w'), ensure_ascii=False)
    print(f"### wrote scores.json (pool_k={pool_k})", flush=True)

    metrics = cov = None
    if os.path.exists(a.gt):
        from eval_submission import score, coverage
        gt = json.load(open(a.gt))
        metrics = score(submission, gt)
        cov = coverage(scores, gt)
        print(f"### metrics {metrics}", flush=True)
        print(f"### coverage {cov}  (pool ceiling for stage-2)", flush=True)

    meta = {"name": a.name, "config": a.config, "checkpoint": a.checkpoint,
            "gallery": a.gallery, "queries": a.queries, "cap_field": cap_field,
            "top_n": a.top_n, "pool_k": pool_k, "top_k": a.top_k,
            "created": datetime.datetime.now().isoformat(timespec='seconds'),
            "metrics": metrics, "coverage": cov}
    json.dump(meta, open(os.path.join(run, 'meta.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"### wrote {run}/ (submission.json, answer.txt, meta.json)", flush=True)


if __name__ == '__main__':
    main()
