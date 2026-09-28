#!/bin/bash
# Base-reconstruction attack: capacity + schedule levers never tested with the S03 recipe.
# Each run = S03 recipe (FTGSPP_LPIPS_W=0.3) on val 001_1, evaluated at 4K solo AND in the P8+P9+J ensemble.
set -u
C=001_1_seq0; R=/home/intern_2603055/vvc/runs/siga_$C; D=/home/intern_2603055/vvc/data/$C
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
run() {  # name  extra-hydra-overrides...
  local n=$1; shift
  local OUT=$R/run_$n
  if [ ! -f $OUT/gaussians.pt ]; then
    while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 30000 ]; do sleep 120; done
    echo "== train $n $(date -u +%H:%M)  [$*]"
    FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
    FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C \
    FTGSPP_POINTS_DIR=$R/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
      /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 \
      train.color_correction=true train.lpips_loss=true "$@" run.output_path=$OUT \
      > /home/intern_2603055/vvc/runs_${C}_$n.log 2>&1 || { echo "$n TRAIN FAILED"; return; }
  fi
  echo -n "$n solo    : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU 2>&1 | grep MEAN
  echo -n "$n +P8P9J  : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU --extra_runs $R/run_P8,$R/run_P9,$R/run_J_dense01 2>&1 | grep MEAN
}
echo "baseline S03 solo 25.517/0.9205/0.2356 ; S03+P8+P9+J 26.107/0.9336/0.2256"
run G3M      init.num_gaussians=3000000
run G4M      init.num_gaussians=4000000
run RELOC50  train.relocation.every=50
echo "CAPACITY PROBE DONE $(date -u +%H:%M)"
