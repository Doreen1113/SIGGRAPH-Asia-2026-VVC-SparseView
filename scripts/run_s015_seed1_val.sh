#!/bin/bash
# S015 second seed on val 001_1 — does a seed ensemble of the best recipe add PSNR on top of P8+P9?
OUT=/home/intern_2603055/vvc/runs/siga_001_1_seq0/run_S015_s1
if [ ! -f $OUT/gaussians.pt ]; then
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 36000 ]; do sleep 120; done
  echo "== train S015 seed1 on 001_1 $(date -u +%m-%d\ %H:%M)"
  FTGSPP_LPIPS_W=0.15 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/001_1_seq0 \
  FTGSPP_POINTS_DIR=/home/intern_2603055/vvc/runs/siga_001_1_seq0/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_001_1_seq0 run.gpu=0 train.seed=1 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_001_1_S015_s1.log 2>&1
fi
R=/home/intern_2603055/vvc/runs/siga_001_1_seq0; C=/home/intern_2603055/vvc/data/001_1_seq0
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 0.5 --every 50 --cull_near_frac 1.1 --cull_min_views 4"
echo -n "S015_s0+P8+P9       : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_S_lpips0.15 $C $CU --extra_runs $R/run_P8,$R/run_P9 2>&1 | grep MEAN
echo -n "S015_s0+s1+P8+P9    : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_S_lpips0.15 $C $CU --extra_runs $R/run_S015_s1,$R/run_P8,$R/run_P9 2>&1 | grep MEAN
echo -n "S015_s0+s1+P8+P9 w2,2,1,1: "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_S_lpips0.15 $C $CU --extra_runs $R/run_S015_s1,$R/run_P8,$R/run_P9 --weights 1,1,1,1 2>&1 | grep MEAN
echo "S015 SEED1 DONE $(date)"
