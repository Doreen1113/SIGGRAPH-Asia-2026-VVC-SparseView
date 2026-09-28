#!/bin/bash
# S015 with 3M Gaussians (default 2M; 1M was worse) — does extra capacity buy PSNR on well-covered views?
OUT=/home/intern_2603055/vvc/runs/siga_001_1_seq0/run_S015_3m
if [ ! -f $OUT/gaussians.pt ]; then
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 32000 ]; do sleep 120; done
  echo "== train S015 3M on 001_1 $(date -u +%m-%d\ %H:%M)"
  FTGSPP_LPIPS_W=0.15 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/001_1_seq0 \
  FTGSPP_POINTS_DIR=/home/intern_2603055/vvc/runs/siga_001_1_seq0/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_001_1_seq0 run.gpu=0 init.num_gaussians=3000000 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_001_1_S015_3m.log 2>&1
fi
R=/home/intern_2603055/vvc/runs/siga_001_1_seq0; C=/home/intern_2603055/vvc/data/001_1_seq0
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 0.5 --every 50 --cull_near_frac 1.1 --cull_min_views 4"
echo -n "S015_3m solo        : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_S015_3m $C $CU 2>&1 | grep MEAN
echo -n "S015_3m+P8+P9       : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_S015_3m $C $CU --extra_runs $R/run_P8,$R/run_P9 2>&1 | grep MEAN
echo "(ref: S015 solo 25.334/0.8956/0.1925 ; S015+P8+P9 25.908/0.9080/0.1877)"
echo "S015 3M DONE $(date)"
