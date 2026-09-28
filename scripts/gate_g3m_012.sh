#!/bin/bash
# Two-scene gate for the 3M-Gaussian capacity gain (the O failure taught us not to trust one scene).
C=012_0_seq0; R=/home/intern_2603055/vvc/runs/siga_$C; D=/home/intern_2603055/vvc/data/$C
OUT=$R/run_G3M
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
if [ ! -f $OUT/gaussians.pt ]; then
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 30000 ]; do sleep 180; done
  echo "== train G3M on 012_0 $(date -u +%H:%M)"
  FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C \
  FTGSPP_POINTS_DIR=$R/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 init.num_gaussians=3000000 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_012_0_G3M.log 2>&1
fi
echo "baseline 012_0: S03+P8+J = 25.027/0.9205/0.2660 (from RESULTS.md, no P9 seed there originally)"
echo -n "012_0 S03 +P8+P9+J : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_S03 $D $CU --extra_runs $R/run_P8,$R/run_P9,$R/run_J 2>&1 | grep MEAN
echo -n "012_0 G3M +P8+P9+J : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT     $D $CU --extra_runs $R/run_P8,$R/run_P9,$R/run_J 2>&1 | grep MEAN
echo "G3M GATE DONE $(date -u +%H:%M)"
