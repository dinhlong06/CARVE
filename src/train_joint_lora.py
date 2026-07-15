"""Joint text+vision LoRA, hybrid loss (deg-image<->noisy-text InfoNCE + frozen-clean anchors).
ITC-only; cross-encoder/itm_head frozen; pose not forwarded. Teachers precomputed.
Single- or multi-GPU (DataParallel splits the batch across GPUs; the contrastive loss
is computed on the gathered full-batch features, so InfoNCE keeps its full negative set).
  python train_joint_lora.py --config configs/ssdc_joint_lora.yaml --checkpoint checkpoint/cmp.pth \
     --teachers data/joint/teachers.pt --output_dir runs/joint_v1 --gpu 4,7
"""
import os, json, argparse, random, time
# Reduce fragmentation OOMs on a contended shared cluster (must precede torch CUDA init).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch, torch.nn as nn, torch.nn.functional as F
from ruamel.yaml import YAML
from transformers import BertTokenizer
import torch.multiprocessing as _mp
from torch.utils.data import DataLoader
from models.model_search import Search
from models.lora_text import apply_joint_lora
from dataset.consistency_dataset import JointDegradeNoisyDataset, collate_joint
from tools.finetune.precompute_teachers import load_teachers
from dataset.utils import pre_caption
_mp.set_sharing_strategy("file_system")   # 64M shm -> file-system sharing


class ITCTowers(nn.Module):
    """forward(imgs, input_ids, attn) -> (i_d, t_n), both L2-normalised [B,2048].
    Pure feature extraction so DataParallel can scatter the batch on dim 0 and
    concatenate the per-shard features back on the primary device. The contrastive
    loss is then computed once, over the FULL gathered batch, in the train loop."""
    def __init__(self, peft_model):
        super().__init__()
        self.m = peft_model

    def forward(self, imgs, input_ids, attn):
        iemb, _ = self.m.get_vision_embeds(imgs)
        i_d = F.normalize(self.m.get_image_feat(iemb), dim=-1)                 # [B,2048]
        temb = self.m.get_text_embeds(input_ids, attn)
        t_n = F.normalize(self.m.get_text_feat(temb), dim=-1)                  # [B,2048]
        return i_d, t_n


def joint_loss(i_d, t_n, ids, teachers, device, temp, w):
    # frozen-clean teachers (cached, row-aligned to this batch)
    i_c = torch.stack([teachers[i][0] for i in ids]).to(device)
    t_c = torch.stack([teachers[i][1] for i in ids]).to(device)
    # primary: symmetric InfoNCE between degraded image and noisy text (full batch)
    logits = (i_d @ t_n.t()) / temp                                           # [B,B]
    tgt = torch.arange(len(ids), device=device)
    L_con = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
    # anchors: perturbed student -> frozen clean teacher
    L_img = (1 - (i_d * i_c).sum(-1)).mean()
    L_txt = (1 - (t_n * t_c).sum(-1)).mean()
    loss = L_con + w['lam_img'] * L_img + w['lam_txt'] * L_txt  # alpha/beta reserved for deferred L_cls/L_tok terms (spec §6) — not used in this ITC-only cut
    return loss, (L_con.item(), L_img.item(), L_txt.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--checkpoint', default='checkpoint/cmp.pth')
    ap.add_argument('--teachers', required=True); ap.add_argument('--output_dir', required=True)
    ap.add_argument('--gpu', default='5'); ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    assert a.gpu != '', "set --gpu"
    # Prefer a CUDA_VISIBLE_DEVICES set by the parent (docker -e): it takes effect before
    # any CUDA context is created. Setting it here is too late once `models` import inits CUDA.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    dev = 'cuda'                                  # logical cuda:0..N-1
    n_gpu = torch.cuda.device_count()
    random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = YAML(typ='safe').load(open(a.config))
    model = Search(config=cfg); model.load_pretrained(a.checkpoint); model = model.to(dev)
    model = apply_joint_lora(model, r=cfg['lora_r'], alpha=cfg['lora_alpha'],
                             dropout=cfg['lora_dropout'], proj_heads=cfg.get('lora_proj_heads', 'none')).to(dev)
    # peft_model keeps the savable reference; towers (optionally DataParallel-wrapped) does the forward.
    peft_model = model
    towers = ITCTowers(peft_model).to(dev)
    runner = nn.DataParallel(towers, device_ids=list(range(n_gpu))) if n_gpu > 1 else towers
    print(f"### training on {n_gpu} GPU(s): CUDA_VISIBLE_DEVICES={a.gpu}", flush=True)
    tok = BertTokenizer.from_pretrained(cfg['text_encoder'])
    teachers = load_teachers(a.teachers)
    ds = JointDegradeNoisyDataset(cfg['train_gallery'], cfg['train_noisy'], cfg['train_img_root'],
                                  cfg['h'], cfg['w'], cfg['max_words'], a.seed,
                                  lhp_p=cfg.get('lhp_p', 0.0),
                                  lhp_scale=cfg.get('lhp_scale', (0.3, 0.7)))
    dl = DataLoader(ds, batch_size=cfg['batch_size_train'], shuffle=True, drop_last=True,
                    num_workers=cfg.get('num_workers', 4), collate_fn=collate_joint)
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad],
                            lr=cfg['optimizer']['lr'], weight_decay=0.01)
    w = {'lam_img': cfg['lam_img'], 'lam_txt': cfg['lam_txt'],
         'alpha': cfg.get('loss_alpha', 0.5), 'beta': cfg.get('loss_beta', 1.0)}
    max_tokens, temp = cfg['max_tokens'], cfg['temp']
    os.makedirs(a.output_dir, exist_ok=True)
    for epoch in range(cfg['schedular']['epochs']):
        t0 = time.time()
        for it, (imgs, caps, ids) in enumerate(dl):
            imgs = imgs.to(dev)
            t = tok([pre_caption(c, max_tokens) for c in caps], padding='max_length',
                    truncation=True, max_length=max_tokens, return_tensors='pt').to(dev)
            i_d, t_n = runner(imgs, t.input_ids, t.attention_mask)   # gathered full batch on cuda:0
            loss, parts = joint_loss(i_d, t_n, ids, teachers, dev, temp, w)
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 50 == 0:
                print(f"ep{epoch} it{it} loss {loss.item():.4f} con {parts[0]:.4f} "
                      f"img {parts[1]:.4f} txt {parts[2]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        peft_model.save_pretrained(os.path.join(a.output_dir, f"adapter_ep{epoch}"))
        print(f"### saved adapter_ep{epoch}", flush=True)


if __name__ == '__main__':
    main()
