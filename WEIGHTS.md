# Weights & Data manifest

What ships in this repo vs what must be downloaded, so the pipeline runs end-to-end.

## ✅ Committed in this repo (small)

| Path | Size | Purpose |
|------|------|---------|
| `weights/adapter_ep1/` | 4.5 MB | joint vision+text LoRA adapter (Stage-1) |
| `weights/lambdamart_vh.lgb` | 16 KB | trained LambdaMART fusion model |
| `configs/`, `src/` | — | code |

**Consequence:** **Stage-2 rerank needs NO GPU and NO model download** once the VLM verdicts are
cached — feed either pool through the s2hy7 stack with the synthetic-selected L5 override
(`src/tools/dump_s2hy7_compliant.py`): base pool → R@1 80.89, **rich pool → R@1 82.10** (best,
compliant). Build the caches once with README §4; every later rerank is CPU-only.

## ⬇️ Must download / host externally (large)

| Artifact | Size | Needed for | How to obtain |
|----------|------|-----------|---------------|
| `cmp.pth` | 884 MB | Stage-1 encode | **host (HF/Drive)** — this repo's release |
| `bert-base-uncased/` | 423 MB | Stage-1 text encoder | HuggingFace `bert-base-uncased` |
| OpenCLIP ViT-H/14 (laion2b) | ~2 GB | regenerate ViT-H sims | HF (via `transformers` CLIPModel) |
| SigLIP so400m | ~1.6 GB | regenerate SigLIP sims | HF `google/siglip-so400m-*` |
| `ext_*/sims_ext.npy` (ViT-H, SigLIP) | 278 MB ea | fusion input | host, **or** regenerate with `src/tools/encode_openclip.py` / `encode_siglip.py` |
| competition gallery+query images | — | Stage-1 encode | AI City 2026 Track-4 test set |
| synthetic train images + `train_queries_noisy.jsonl` | 44 MB + imgs | **retrain** LoRA / LambdaMART | competition synthetic set |
| Qwen3-VL-8B / InternVL3.5 / SmolVLM2 | large | build the Stage-2 verify caches | HF — *needed once; CPU-only thereafter* |

## Layout expected at run time (symlink or copy the downloads here)

```
weights/cmp.pth
weights/bert-base-uncased/
data/gallery.jsonl  data/queries_competition.jsonl
data/ext_vith/sims_ext.npy  data/ext_siglip/sims_ext.npy
src/submissions/            # <- verify caches + fusion pools (build: README §4)
```
