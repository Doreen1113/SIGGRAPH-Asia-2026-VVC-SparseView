#!/bin/bash
# Train the J recipe at full training resolution (scale 1.0) on the 5 test cases, as ensemble member O.
set -u
export FTGSPP_TRAIN_EVAL_INTERVAL=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  OUT=/home/intern_2603055/vvc/runs/siga_$c/run_O_s0
  [ -f $OUT/gaussians.pt ] && continue
  echo "== train O $c $(date)"
  FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$c \
  FTGSPP_POINTS_DIR=/home/intern_2603055/vvc/runs/siga_$c/points_stride10 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$c run.gpu=0 data.scale=1.0 \
    train.color_correction=true train.lpips_loss=true train.iterations=15000 run.output_path=$OUT > /home/intern_2603055/vvc/runs_${c}_O_s0.log 2>&1
done
echo "RECIPE O DONE $(date)"
