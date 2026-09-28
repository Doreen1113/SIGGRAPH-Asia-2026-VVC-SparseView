#!/bin/bash
# Capacity and init are independent axes and BOTH passed. Test the combination (gains are not assumed additive).
C=001_1_seq0; R=/home/intern_2603055/vvc/runs/siga_$C; D=/home/intern_2603055/vvc/data/$C
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
PTS=$R/points_dense200k
for i in $(seq 1 60); do [ -d $PTS ] && break; sleep 120; done
[ -d $PTS ] || { echo "dense points not found at $PTS - peer may use another path"; ls $R | grep points; exit 1; }
OUT=$R/run_G4M_DENSE
if [ ! -f $OUT/gaussians.pt ]; then
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 30000 ]; do sleep 180; done
  echo "== train G4M+DENSEPTS $(date -u +%H:%M)"
  FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C FTGSPP_POINTS_DIR=$PTS FTGSPP_TRAIN_EVAL_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 \
    init.num_gaussians=4000000 init.num_points_per_frame=200000 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_001_1_G4M_DENSE.log 2>&1
fi
echo "refs: S03+P8P9J 26.107 | G4M 26.373 | DENSEPTS 26.219"
echo -n "G4M+DENSE solo   : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU 2>&1 | grep MEAN
echo -n "G4M+DENSE +P8P9J : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU --extra_runs $R/run_P8,$R/run_P9,$R/run_J_dense01 2>&1 | grep MEAN
echo "COMBO DONE $(date -u +%H:%M)"
