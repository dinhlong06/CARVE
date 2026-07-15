# CARVE — Consensus-Aware Retrieve-and-Verify Ensemble

**Text-based Person Anomaly Search** — AI City 2026 Track-4 / ECCV.

Clean, self-contained package to **reproduce, train, and evaluate** our best submission. Isolated
from the working repos — no stale code or tuned state.

CARVE is a two-stage cascade whose every accuracy lever is built on **cross-model agreement**: a
learned fusion of three retrieval encoders proposes candidates, and a panel of three
different-family multimodal LLMs verifies them. Agreement is domain-invariant, which is what lets a
**synthetic-only**-trained pipeline transfer to the real gallery.

- **Best result:** local-proxy **R@1 82.10 / mAP 87.99** — rich-feature LambdaMART stage-1 + s2hy7
  with a default blend and a synthetic-selected L5 override. Fully compliant (no test data / `gt.json`
  in training or selection). See [`RESULTS.md`](RESULTS.md) → "Rich fusion features" and "Compliant
  Stage-2 selection". Leaderboard filled after upload.
- **Compliance:** Track-4 synthetic-data-only. No test data / `gt.json` touches training or
  hyper-parameter selection. Fusion is a synthetic-trained **LambdaMART**; Stage-2 uses **default**
  hyper-parameters. No gt-tuned "diagnostic" numbers are reported.
- **Method & design rationale:** [`SOLUTION.md`](SOLUTION.md). **Experimental results:**
  [`RESULTS.md`](RESULTS.md). **Weights/data manifest:** [`WEIGHTS.md`](WEIGHTS.md).

## Pipeline
```
query ─▶ CMP (Swin+BERT) + joint LoRA ─┐
                                        ├▶ LambdaMART fusion ─topK─▶ Stage-2 VLM verify ─▶ ranking
gallery ─▶ + ViT-H + SigLIP ────────────┘   (LightGBM)              (Qwen3-VL + InternVL + SmolVLM,
                                                                     hybrid rerank, default hp)
```

---

## 1. Environment setup

Two Docker images (Stage-1 and Stage-2 have incompatible deps). Build from `deps/`:

```bash
docker build -f deps/Dockerfile.stage1 -t ssdc-stage1 .   # transformers 4.44 (encode, LoRA, fusion)
docker build -f deps/Dockerfile.stage2 -t ssdc-stage2 .   # vLLM (Stage-2 VLM verify)
# or: docker compose -f deps/docker-compose.yml up -d
```
Python deps also in `deps/requirements.stage1_*.txt`. Stage-2 **rerank** (replayed from the cached VLM
verdicts) is pure-Python/CPU and needs only `numpy` — no Docker required.


## 2. Data preparation & augmentation (build LoRA train set + eval set)

All synthetic-only. Toolkit in `src/tools/synthetic_eval/` (run each with `--help` / see its
docstring for full flags; commands below are the exact arg names):

```bash
# a) split the synthetic annotation by source_id (train / held-out val)
python src/tools/synthetic_eval/make_split.py \
  --ann data/synthetic_eval/annotation.json --val 0.02 --out data/synthetic_eval/split_ids.json

# b) LLM style-rewrite clean captions -> NOISY query candidates (Qwen/vLLM), easy->hard variants
python src/tools/synthetic_eval/gen_noisy_val.py \
  --clean data/synthetic_eval/train_queries_clean.jsonl \
  --out   data/synthetic_eval/train_queries_noisy_candidates.jsonl --variants 3

# c) meaning-gate: keep a variant only if CMP-text cosine to its clean caption >= 0.85
python src/tools/synthetic_eval/filter_noisy.py \
  --config configs/ssdc_openpose_ver2.yaml --checkpoint weights/cmp.pth \
  --clean      data/synthetic_eval/train_queries_clean.jsonl \
  --candidates data/synthetic_eval/train_queries_noisy_candidates.jsonl \
  --out        data/synthetic_eval/train_queries_noisy.jsonl --gate 0.85

# d) test-realistic gallery degradation (light crop + low-res + JPEG)
python src/tools/synthetic_eval/dump_degraded_gallery.py \
  --gallery data/synthetic_eval/val_gallery.jsonl --src-root dataset \
  --out-dir dataset/val_gallery_degraded --manifest data/synthetic_eval/val_gallery_degraded.jsonl

# e) eval on the synthetic val split
python src/tools/synthetic_eval/run_eval.py \
  --config configs/ssdc_openpose_ver2.yaml --checkpoint weights/cmp.pth \
  --eval-dir data/synthetic_eval --image-root data/PAB/ --split noisy
```
Output feeds `train_joint_lora.py` (train set) and LambdaMART fusion selection (val-hard).

## 3. Reproduce inference

**Best (compliant) — offline, CPU-only, from the cached VLM verdicts** (verified R@1 82.10):
```bash
cd src/submissions && python3 ../tools/synth_l5_select.py   # synthetic L5 selection -> l5_selected.json
cd .. && python3 tools/dump_s2hy7_compliant.py              # rich pool + default blend + synth-L5
python3 eval_submission.py submissions/lambdamart_3enc_rich_s2hy7_synL5
# -> R@1 82.10 / R@5 94.74 / R@10 95.90 / mAP 87.99   (L5 fired 84)
```
That run directory gets the uploadable `answer.txt` (1,978 lines × top-10 tokens).

**Full chain from scratch** (needs GPU + `cmp.pth`, see `WEIGHTS.md`), pinned in `scripts/run_all.sh`:
Stage-1 encode (`infer_submit.py --joint_adapter weights/adapter_ep1 --blank_pose --pool_k 128`) →
LambdaMART fusion → Stage-2. Anchor: `base_pose` reproduces R@1 69.51 exactly.

## 4. Retrain (optional — shipped weights already included)

`scripts/train.sh` — synthetic-only:
1. **Joint LoRA:** `train_joint_lora.py` (epoch selected on synthetic val-hard; we ship ep1).
2. **Ext caches:** `encode_openclip.py` (ViT-H/14 laion2b), `encode_siglip.py` (so400m).
3. **LambdaMART:** `lambdamart_transfer.py` (`--with_sig` = 3-encoder); trains on synthetic, transfers.
   → writes `weights/lambdamart_vh.lgb`. **CPU**, no GPU.
Stage-2 needs no training — but the verify caches **can be regenerated from the VLMs** (below).

### Regenerate the Stage-2 verify caches (best baseline)
Run inside the Stage-2 vLLM image, cwd `src/`. Decoding is greedy (`temperature=0`, first-token
`p(Yes)` logprob) → **deterministic**. Turing GPUs: keep `SSDC_ATTN_BACKEND=TRITON_ATTN` (default;
FlashInfer crashes silently → all-"No"); always check the cache Yes-rate > 0.

Models (HuggingFace): Qwen3-VL-8B-Instruct, InternVL3.5, SmolVLM2.
**GPU sizing (validated):** fp16 Qwen3-VL-8B (~17 GB) needs `SSDC_TP_SIZE=2` across two GPUs; the
**AWQ-4bit** build (~5 GB) runs `SSDC_TP_SIZE=1` on one 11 GB card. Set `SSDC_GPU_UTIL` just under
the card's free memory (e.g. 0.7–0.9). Verified: regenerating a 22-query slice with AWQ matches the
our reference cache (median|Δp|=0.000). On Turing (cc<8), FA2 is unsupported → `TRITON_ATTN` (default)
+ tiny numeric drift, rankings unaffected.
```bash
# Qwen3-VL — binary Yes/No + continuous p(Yes)
python3 vllm_infer_SSDC.py --run fuse_cmp512_vith_siglip_w03 --round 20 \
  --model_dir /path/Qwen3-VL-8B-Instruct --name s2cache_w03_r20
python3 tools/s2_logprob_verify.py --run fuse_gated_best \
  --ext_run fuse_cmp512_vith_siglip_recallmax_w13 --ext_k 15 --round 10 \
  --model_dir /path/Qwen3-VL-8B-Instruct --name s2cache_logprob_fp16_gr15
# InternVL3.5 + SmolVLM2 — different-family verifiers, over the candidate pairs
python3 tools/s2_pilot_verify.py --parse logprob \
  --model_dir /path/InternVL3_5 --pairs submissions/ivl_full_pairs.json --name s2cache_internvl35
python3 tools/s2_pilot_verify.py --parse logprob \
  --model_dir /path/SmolVLM2   --pairs submissions/smo_full_pairs.json --name s2cache_smolvlm2
```
Then the §3 compliant reproduce should print **R@1 82.10** — end-to-end proof the VLM step reproduced.


## 5. Repo layout
```
CARVE/
├── README.md  SOLUTION.md  RESULTS.md  WEIGHTS.md
├── configs/                 backbone + joint-LoRA yaml/json
├── deps/                    Dockerfile.stage1/2, docker-compose, requirements
├── weights/  adapter_ep1/ (LoRA)  lambdamart_vh.lgb   [cmp.pth = external]
├── scripts/  run_all.sh (inference)  train.sh (retrain)
└── src/
    ├── infer_submit.py  eval_submission.py  train_joint_lora.py  vllm_infer_SSDC.py
    ├── models/  datasets/  dataset/          backbone + data loaders
    ├── tools/   fuse_gated  lambdamart_*  encode_openclip/siglip  s2*  synthetic_eval/  finetune/
    └── submissions/  verify caches, fusion pools, run outputs
```

## Integrity
- Code verified complete: 0 unresolved local imports, all files byte-compile.
- Stage-2 endpoint reproduces offline from the cached VLM verdicts (md5-verified vs originals).

