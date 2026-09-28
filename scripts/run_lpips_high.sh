#!/bin/bash
# LPIPS weight sweep beyond 0.15 — LPIPS decides the rank and the 0.01->0.15 trend was non-monotonic.
for W in 0.3 0.5; do
  OUT=/home/intern_2603055/vvc/runs/siga_001_1_seq0/run_S$W
  [ -f $OUT/gaussians.pt ] && continue
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 34000 ]; do sleep 120; done
  echo "== train S$W on 001_1 $(date -u +%m-%d\ %H:%M)"
  FTGSPP_LPIPS_W=$W FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/001_1_seq0 \
  FTGSPP_POINTS_DIR=/home/intern_2603055/vvc/runs/siga_001_1_seq0/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_001_1_seq0 run.gpu=0 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_001_1_S$W.log 2>&1
  R=/home/intern_2603055/vvc/runs/siga_001_1_seq0; C=/home/intern_2603055/vvc/data/001_1_seq0
  EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
  echo -n "S$W solo 4K   : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $C $CU 2>&1 | grep MEAN
  echo -n "S$W +P8+P9 4K : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $C $CU --extra_runs $R/run_P8,$R/run_P9 2>&1 | grep MEAN
done
echo "LPIPS HIGH SWEEP DONE $(date)"
