import json, random
import torch
from torch.utils.data import Dataset
from dataset.utils import pre_caption
from tools.finetune.augment_query import augment, CHANGE_TYPES


def _load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class ConsistencyTrainDataset(Dataset):
    """Text-only. Each item: (clean caption, one synthetic perturbed view, change type)."""
    def __init__(self, caption_files, max_words, seed=42):
        self.max_words = max_words
        self.ann = []
        for f in caption_files:
            for rec in _load_jsonl(f):
                if rec.get("caption"):
                    self.ann.append(rec["caption"])
        self.base_seed = seed
        print(f"ConsistencyTrainDataset: {len(self.ann)} captions from {len(caption_files)} files")

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        rng = random.Random(self.base_seed * 1_000_003 + index)
        clean = pre_caption(self.ann[index], self.max_words)
        change = rng.choice(CHANGE_TYPES)
        pert = pre_caption(augment(clean, change, rng), self.max_words)
        return clean, pert, change


def collate_consistency(batch):
    clean = [b[0] for b in batch]
    pert = [b[1] for b in batch]
    change = [b[2] for b in batch]
    return clean, pert, change


class LLMAugConsistencyDataset(Dataset):
    """Pre-generated LLM (clean, perturbed, change) triples from tools/finetune/llm_augment.py.
    Each jsonl line: {"clean": <orig caption>, "caption": <Qwen3-perturbed>, "change": <type>}.
    Unlike ConsistencyTrainDataset, perturbations are NOT synthesized on-the-fly via augment() —
    they are realistic LLM rewrites whose cos(clean,pert) matches the true competition gap (~0.86),
    which deterministic augment (~0.99) could not reproduce."""
    def __init__(self, pair_files, max_words):
        self.max_words = max_words
        self.pairs = []
        for f in pair_files:
            for rec in _load_jsonl(f):
                clean, pert = rec.get("clean"), rec.get("caption")
                if clean and pert:
                    self.pairs.append((clean, pert, rec.get("change", "all")))
        print(f"LLMAugConsistencyDataset: {len(self.pairs)} pairs from {len(pair_files)} files")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        clean, pert, change = self.pairs[index]
        return pre_caption(clean, self.max_words), pre_caption(pert, self.max_words), change


import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class ConsistencyValDataset(Dataset):
    """Drop-in for eval.evaluation_itc: gallery images + perturbed queries with change labels."""
    def __init__(self, gallery_jsonl, queries_jsonl, img_root, h, w, max_words=56):
        self.img_root = img_root
        norm = transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))
        self.transform = transforms.Compose([transforms.Resize((h,w), interpolation=InterpolationMode.BICUBIC),
                                             transforms.ToTensor(), norm])
        self.gallery = _load_jsonl(gallery_jsonl)
        self.g_pids = [r["image_id"] for r in self.gallery]
        q = _load_jsonl(queries_jsonl)
        self.text = [pre_caption(r["text"], max_words) for r in q]
        self.q_pids = [r["image_id"] for r in q]
        self.q_changes = [r["change"] for r in q]

    def __len__(self):
        return len(self.gallery)

    def __getitem__(self, index):
        base = os.path.basename(self.gallery[index]["image"])
        image = self.transform(Image.open(os.path.join(self.img_root, base)).convert("RGB"))
        pose = self.transform(Image.open(os.path.join(self.img_root, "pose", base)).convert("RGB"))
        return image, pose, index


import io
import math
from collections import defaultdict
from tools.synthetic_eval.degrade import degrade


def _lhp_local_crop(img, scale):
    """HUI idea-1 'local view': random crop of 30-70% area (aspect 3/4..4/3), any position.
    Unlike degrade()'s light center crop this deliberately loses context to force the
    encoder to discriminate on person-level details (the s2hy5 hard-core failure mode)."""
    W, H = img.size
    for _ in range(10):
        s = random.uniform(*scale)
        ar = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
        cw, ch = int(round(math.sqrt(W * H * s * ar))), int(round(math.sqrt(W * H * s / ar)))
        if 0 < cw <= W and 0 < ch <= H:
            x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch)
            return img.crop((x0, y0, x0 + cw, y0 + ch))
    return img


class JointDegradeNoisyDataset(Dataset):
    """One item per unique image_id: (degraded image, a random noisy variant, image_id)."""
    def __init__(self, gallery_jsonl, noisy_jsonl, img_root, h=224, w=224, max_words=56, seed=42,
                 lhp_p=0.0, lhp_scale=(0.3, 0.7)):
        self.img_root, self.max_words, self.seed = img_root, max_words, seed
        self.lhp_p, self.lhp_scale = float(lhp_p), tuple(lhp_scale)
        norm = transforms.Normalize((0.48145466,0.4578275,0.40821073),
                                    (0.26862954,0.26130258,0.27577711))
        self.post = transforms.Compose([transforms.Resize((h,w), interpolation=InterpolationMode.BICUBIC),
                                        transforms.ToTensor(), norm])
        self.img = {r["image_id"]: r["image"] for r in _load_jsonl(gallery_jsonl)}
        noisy = defaultdict(list)
        for r in _load_jsonl(noisy_jsonl):
            if r["image_id"] in self.img and r.get("caption"):
                noisy[r["image_id"]].append(r["caption"])
        self.ids = sorted(noisy.keys())
        self.noisy = noisy
        print(f"JointDegradeNoisyDataset: {len(self.ids)} unique images, "
              f"{sum(len(v) for v in self.noisy.values())} noisy variants")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        iid = self.ids[index]
        rel = self.img[iid]
        img = Image.open(os.path.join(self.img_root, rel)).convert("RGB")   # rel may be nested (train/imgs_N/...)
        if self.lhp_p > 0 and random.random() < self.lhp_p:
            img = _lhp_local_crop(img, self.lhp_scale)               # local view; global = no-crop branch
        deg, q = degrade(img, iid)                                   # non-square low-res PIL + jpeg q
        # bake jpeg artifacts in-memory, then the CMP square-resize warps it once (real distortion)
        buf = io.BytesIO(); deg.save(buf, format="JPEG", quality=q); buf.seek(0)
        img_t = self.post(Image.open(buf).convert("RGB"))
        cap = pre_caption(random.choice(self.noisy[iid]), self.max_words)
        return img_t, cap, iid


def collate_joint(batch):
    imgs = torch.stack([b[0] for b in batch])
    caps = [b[1] for b in batch]
    ids = [b[2] for b in batch]
    return imgs, caps, ids
