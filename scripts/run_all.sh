#!/usr/bin/env bash
# Reproduce the full ablation (A0..A4 Stage-1, B1..B5 Stage-2) from one fixed setup.
# Pin every hyperparameter here so numbers are reproducible. Run from repo root inside the
# Stage-1 Docker image (Stage-2 steps need the Stage-2/vLLM image or hosted verify caches).
set -euo pipefail

GAL=data/gallery.jsonl                 # competition gallery (36,773)
QRY=data/queries_competition.jsonl     # perturbed queries (1,978)
CKPT=weights/cmp.pth
ADAPTER=weights/adapter_ep1
POOL=128
GPU=${1:-0}
POSE="--blank_pose"                    # competitive chain: blank pose throughout

# ---- Stage 1 ----
# A0 base CMP (blank pose)  [anchor: pose-on reproduces 69.51, blank 69.11]
python src/infer_submit.py --name A0_base_cmp   --config configs/ssdc_joint_lora.yaml \
  --checkpoint $CKPT --gallery $GAL --queries $QRY --pool_k $POOL $POSE --gpu $GPU

# A1 + joint LoRA
python src/infer_submit.py --name A1_joint_lora --config configs/ssdc_joint_lora.yaml \
  --checkpoint $CKPT --joint_adapter $ADAPTER --gallery $GAL --queries $QRY \
  --pool_k $POOL $POSE --gpu $GPU

# A2 + ViT-H  |  A3 + SigLIP  |  A4 + both   (gated-RRF; DEFAULT gate, pinned below)
WLO=0.0; WHI=0.15; GAMMA=1.5   # <-- pinned default gate (document; not gt-tuned)
python src/tools/fuse_gated.py --cmp submissions/A1_joint_lora/scores.json \
  --ext data/ext_vith/sims_ext.npy \
  --w_lo_grid $WLO --w_hi_grid $WHI --gamma_grid $GAMMA --write_best_r1 A2_vith
python src/tools/fuse_gated.py --cmp submissions/A1_joint_lora/scores.json \
  --ext data/ext_siglip/sims_ext.npy \
  --w_lo_grid $WLO --w_hi_grid $WHI --gamma_grid $GAMMA --write_best_r1 A3_siglip
python src/tools/fuse_gated.py --cmp submissions/A1_joint_lora/scores.json \
  --ext data/ext_vith/sims_ext.npy --ext data/ext_siglip/sims_ext.npy \
  --w_lo_grid $WLO --w_hi_grid $WHI --gamma_grid $GAMMA --write_best_r1 A4_fusion

# ---- Stage 2 ----  (reuse hosted verify caches on the A4 top-K, or regenerate — see README)
# B1..B5 incremental rungs; s2hy6_lab.py 'all' emits the full ladder incl. B5 (L5 override).
python src/tools/s2hy6_lab.py all

echo "Done. Ladder printed above. Final compliant ranking + leaderboard answer.txt:"
echo "  python src/tools/dump_s2hy7_compliant.py"
echo "  -> src/submissions/lambdamart_3enc_rich_s2hy7_synL5/  ; fill LB columns in README.md"
