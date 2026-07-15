"""Cache frozen clean teacher embeds (adapter-off base) for the joint-LoRA consistency anchors.
  python tools/finetune/precompute_teachers.py --config configs/ssdc_joint_lora.yaml \
     --ckpt checkpoint/cmp.pth --gpu <id> --out data/joint/teachers.pt
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch, torch.nn.functional as F
from ruamel.yaml import YAML
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import BertTokenizer
from models.model_search import Search
from dataset.utils import pre_caption

def load_teachers(path):
    d = torch.load(path, map_location='cpu')
    return {i: (d["i_feat"][k], d["t_feat"][k]) for k, i in enumerate(d["ids"])}

def _img_tf(h, w):
    norm = transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))
    return transforms.Compose([transforms.Resize((h,w), interpolation=InterpolationMode.BICUBIC),
                               transforms.ToTensor(), norm])

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--ckpt', default='checkpoint/cmp.pth')
    ap.add_argument('--gallery', default='data/synthetic_eval/train_gallery.jsonl')
    ap.add_argument('--clean', default='data/synthetic_eval/train_queries_clean.jsonl')
    ap.add_argument('--img-root', default='data/PAB')      # dataset/train mounted here in ssdc-eval
    ap.add_argument('--noisy', default='data/synthetic_eval/train_queries_noisy.jsonl',
                    help='restrict teachers to image_ids that survived the meaning-gate')
    ap.add_argument('--out', default='data/joint/teachers.pt')
    ap.add_argument('--gpu', default='0'); ap.add_argument('--bs', type=int, default=64)
    a = ap.parse_args()
    assert a.gpu != '', "set --gpu (non-empty CUDA device)"
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu; dev = 'cuda'
    cfg = YAML(typ='safe').load(open(a.config))
    model = Search(config=cfg); model.load_pretrained(a.ckpt); model = model.to(dev).eval()
    tok = BertTokenizer.from_pretrained(cfg['text_encoder']); mw = cfg.get('max_words', 56)
    tf = _img_tf(cfg['h'], cfg['w'])

    keep = {json.loads(l)["image_id"] for l in open(a.noisy) if l.strip()}
    img_of = {r["image_id"]: r["image"] for r in (json.loads(l) for l in open(a.gallery) if l.strip())}
    cap_of = {r["image_id"]: r["caption"] for r in (json.loads(l) for l in open(a.clean) if l.strip())}
    ids = sorted(i for i in keep if i in img_of and i in cap_of)
    print(f"teachers for {len(ids)} images", flush=True)

    i_feats, t_feats = [], []
    for s in range(0, len(ids), a.bs):
        chunk = ids[s:s+a.bs]
        imgs = torch.stack([tf(Image.open(os.path.join(a.img_root, img_of[i])).convert("RGB")) for i in chunk]).to(dev)
        emb, _ = model.get_vision_embeds(imgs)
        i_feats.append(F.normalize(model.get_image_feat(emb), dim=-1).cpu())
        t = tok([pre_caption(cap_of[i], mw) for i in chunk], padding='max_length',
                truncation=True, max_length=mw, return_tensors='pt').to(dev)
        t_feats.append(F.normalize(model.get_text_feat(model.get_text_embeds(t.input_ids, t.attention_mask)), dim=-1).cpu())
        if (s // a.bs) % 20 == 0: print(f"  {s+len(chunk)}/{len(ids)}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"ids": ids, "i_feat": torch.cat(i_feats), "t_feat": torch.cat(t_feats)}, a.out)
    print(f"saved -> {a.out}", flush=True)

if __name__ == '__main__':
    main()
