#!/usr/bin/env bash
# Reproduce the trained components from scratch (synthetic-data-only, Track-4 compliant).
# Only needed if you want to RE-TRAIN; to just run inference, use the shipped
# weights/adapter_ep1 + weights/lambdamart_vh.lgb. Run inside the Stage-1 Docker image.
set -euo pipefail
GPU=${1:-0}

# 1) Joint vision+text LoRA on synthetic (produces runs/joint_v1/adapter_epN; we ship ep1)
python src/train_joint_lora.py --config configs/ssdc_joint_lora.yaml \
  --checkpoint weights/cmp.pth --output_dir runs/joint_v1 --gpu $GPU
#    epoch selection is on synthetic val-hard (see docs); we use ep1.

# 2) External encoder similarity caches (ViT-H, SigLIP) on gallery+queries
python src/tools/encode_openclip.py --gallery data/gallery.jsonl --queries data/queries_competition.jsonl \
  --name ext_vith --gpu $GPU
python src/tools/encode_siglip.py --model <siglip-so400m> \
  --gallery data/gallery.jsonl --queries data/queries_competition.jsonl --name ext_siglip --gpu $GPU

# 3) LambdaMART fusion — train on synthetic val-hard, transfer to localeval (CPU, no GPU)
#    default = CMP+ViT-H ; --with_sig = CMP+ViT-H+SigLIP (the deployed 3-encoder model)
python src/tools/lambdamart_transfer.py --with_sig \
  --le_cmp submissions/A1_joint_lora/scores.json \
  --dump_pool fusion_pool/lambdamart_pool_3enc_scores.json
#    -> also writes weights/lambdamart_vh.lgb

echo "Training done. Stage-2 needs no training (uses committed verify caches)."
