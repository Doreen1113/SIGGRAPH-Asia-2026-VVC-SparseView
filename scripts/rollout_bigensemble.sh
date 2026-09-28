#!/bin/bash
# Rank-1.50 path: add two structurally-different members (marginal_gating, 4M capacity) to all 5 test cases,
# giving the +0.001 SSIM headroom that lets stronger Difix flip FULL-LPIPS while holding SSIM rank-1.
# Two-scene gated: 001_1 +0.0015 SSIM/+0.62 dB, 012_0 +0.0021 SSIM/+0.50 dB.
set -u
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
train() { # name case overrides...
  local n=$1 C=$2; shift 2
  local R=/home/intern_2603055/vvc/runs/siga_$C; local OUT=$R/run_$n
  [ -f $OUT/gaussians.pt ] && { echo "skip $n/$C (done)"; return; }
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 30000 ]; do sleep 180; done
  echo "== train $n $C $(date -u +%m-%d\ %H:%M)"
  FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 \
    train.color_correction=true train.lpips_loss=true "$@" run.output_path=$OUT \
    > /home/intern_2603055/vvc/runs_${C}_$n.log 2>&1 || echo "$n/$C FAILED"
}
for C in $CASES; do train GATE_s0 $C model.marginal_gating=true; done
for C in $CASES; do train G4M_s0  $C init.num_gaussians=4000000; done
echo "BIG ENSEMBLE TRAINING DONE $(date -u +%m-%d\ %H:%M)"
