#!/bin/bash
# S = J recipe with a higher training LPIPS weight (0.05). Replaces J as the ensemble's primary member.
export FTGSPP_TRAIN_EVAL_INTERVAL=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  OUT=/home/intern_2603055/vvc/runs/siga_$c/run_S_s0
  [ -f $OUT/gaussians.pt ] && continue
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 34000 ]; do sleep 120; done
  echo "== train S $c $(date -u +%m-%d\ %H:%M)"
  FTGSPP_LPIPS_W=0.05 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$c \
  FTGSPP_POINTS_DIR=/home/intern_2603055/vvc/runs/siga_$c/points_stride10 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$c run.gpu=0 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_${c}_S_s0.log 2>&1
done
echo "S CHAIN DONE $(date)"
