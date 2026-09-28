#!/bin/bash
# Structural hypothesis: we train at data.scale=0.5 but are evaluated at native 4K. The model never sees
# detail finer than 2K, which structurally caps LPIPS/SSIM and the foreground metrics.
# Test full-resolution training as the PRIMARY model with the current S03 recipe, judged on FULL *and* FG.
C=001_1_seq0; R=/home/intern_2603055/vvc/runs/siga_$C; D=/home/intern_2603055/vvc/data/$C
OUT=$R/run_FULLRES
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
if [ ! -f $OUT/gaussians.pt ]; then
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 26000 ]; do sleep 180; done
  echo "== train FULLRES (data.scale=1.0, S03 recipe) $(date -u +%H:%M)"
  FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 data.scale=1.0 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT \
    > /home/intern_2603055/vvc/runs_${C}_FULLRES.log 2>&1 || { echo FULLRES FAILED; exit 1; }
fi
echo "baseline S03 solo 25.517/0.9205/0.2356 ; +P8P9J 26.107/0.9336/0.2256"
echo -n "FULLRES solo    : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU 2>&1 | grep MEAN
echo -n "FULLRES +P8P9J  : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU --extra_runs $R/run_P8,$R/run_P9,$R/run_J_dense01 2>&1 | grep MEAN
echo "== FG metrics (the half of the score we are losing) =="
FGDIAG_MAIN=run_FULLRES /home/intern_2603055/vvc/ftg.sh /tmp/claude-1001/fg_diagnose.py $C /tmp/claude-1001/fgdiag_fullres 2>&1 | grep -E "^[0-9]+:"
echo "FULLRES TEST DONE $(date -u +%H:%M)"
