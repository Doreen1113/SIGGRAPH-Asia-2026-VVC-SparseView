#!/bin/bash
# Validate the S recipe (higher training LPIPS weight) on the SECOND val scene before trusting it.
OUT=/home/intern_2603055/vvc/runs/siga_012_0_seq0/run_S
if [ ! -f $OUT/gaussians.pt ]; then
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 36000 ]; do sleep 120; done
  echo "== train S on 012_0 $(date -u +%m-%d\ %H:%M)"
  FTGSPP_LPIPS_W=0.05 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/012_0_seq0 \
  FTGSPP_POINTS_DIR=/home/intern_2603055/vvc/runs/siga_012_0_seq0/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_012_0_seq0 run.gpu=0 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_012_0_S.log 2>&1
fi
R=/home/intern_2603055/vvc/runs/siga_012_0_seq0; C=/home/intern_2603055/vvc/data/012_0_seq0
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 0.5 --every 50 --cull_near_frac 1.1 --cull_min_views 4"
echo -n "012_0 S solo : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_S $C $CU 2>&1 | grep MEAN
echo -n "012_0 J+P8   : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_J $C $CU --extra_runs $R/run_P8 2>&1 | grep MEAN
echo -n "012_0 S+P8   : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_S $C $CU --extra_runs $R/run_P8 2>&1 | grep MEAN
echo "S VALIDATION DONE $(date)"
