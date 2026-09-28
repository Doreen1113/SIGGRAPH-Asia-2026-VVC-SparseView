#!/bin/bash
# CRITICAL PATH to rank 1.500 (outright first): full-resolution training as the primary ensemble member.
# Measured on val: FULL-LPIPS -0.0073 and FG-LPIPS -0.0067 at ~zero SSIM cost -- the only change found that
# moves the SSIM/LPIPS curve outward instead of sliding along it.
for C in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  R=/home/intern_2603055/vvc/runs/siga_$C; OUT=$R/run_FULLRES_s0
  [ -f $OUT/gaussians.pt ] && { echo "skip $C"; continue; }
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 26000 ]; do sleep 180; done
  echo "== train FULLRES $C $(date -u +%m-%d\ %H:%M)"
  FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 data.scale=1.0 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT \
    > /home/intern_2603055/vvc/runs_${C}_FULLRES_s0.log 2>&1 || echo "FULLRES $C FAILED"
done
echo "FULLRES ROLLOUT DONE $(date -u +%m-%d\ %H:%M)"
