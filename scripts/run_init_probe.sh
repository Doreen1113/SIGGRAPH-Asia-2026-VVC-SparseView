#!/bin/bash
# Base-reconstruction attack: initialization density/scale levers never tested with the S03 recipe.
# Each run = S03 recipe (FTGSPP_LPIPS_W=0.3) on val 001_1, evaluated at 4K solo AND in the P8+P9+J ensemble.
set -u
C=001_1_seq0; R=/home/intern_2603055/vvc/runs/siga_$C; D=/home/intern_2603055/vvc/data/$C
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
run() {  # name  points_path  extra-hydra-overrides...
  local n=$1; local pdir=$2; shift 2
  local OUT=$R/run_$n
  if [ ! -f $OUT/gaussians.pt ]; then
    while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 30000 ]; do sleep 120; done
    echo "== train $n $(date -u +%H:%M)  [$*]"
    FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
    FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C \
    FTGSPP_POINTS_DIR=$pdir FTGSPP_TRAIN_EVAL_INTERVAL=0 \
      /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 \
      train.color_correction=true train.lpips_loss=true init.points_path=$pdir "$@" run.output_path=$OUT \
      > /home/intern_2603055/vvc/runs_${C}_$n.log 2>&1 || { echo "$n TRAIN FAILED"; return; }
  fi
  echo -n "$n solo    : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU 2>&1 | grep MEAN
  echo -n "$n +P8P9J  : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU --extra_runs $R/run_P8,$R/run_P9,$R/run_J_dense01 2>&1 | grep MEAN
}
echo "baseline S03 solo 25.517/0.9205/0.2356 ; S03+P8+P9+J 26.107/0.9336/0.2256"
run DENSEPTS  $R/points_dense200k  init.num_points_per_frame=200000
run STRIDE5   $R/points_stride5    init.keyframe_stride=5
run SCALE05   $R/points_stride10  init.scale=0.05
echo "INIT PROBE DONE $(date -u +%H:%M)"
