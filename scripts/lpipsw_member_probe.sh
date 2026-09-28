#!/bin/bash
# Reconstruction-level probe: the only class of change that has ever shifted the LPIPS/SSIM curve was training-side
# (full-res). Try a heavier LPIPS loss at full-res (0.6 vs S03's 0.3) as an ensemble member. Sequential, no gating.
C=001_1_seq0; R=/home/intern_2603055/vvc/runs/siga_$C; D=/home/intern_2603055/vvc/data/$C
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
OUT=$R/run_FULLRES_LP06
if [ ! -f $OUT/gaussians.pt ]; then
  echo "== train FULLRES_LP06 (data.scale=1.0, LPIPS_W 0.6) $(date -u +%H:%M)"
  FTGSPP_LPIPS_W=0.6 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 data.scale=1.0 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_${C}_FULLRES_LP06.log 2>&1 || { echo FAILED; exit 1; }
fi
SIX="$R/run_P8,$R/run_P9,$R/run_J_dense01,$R/run_GATEONLY,$R/run_G4M"
echo "ref 6-member (FULLRES+5): 26.528/0.9348/0.2250"
echo -n "LP06 solo                : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU 2>&1 | grep MEAN
echo -n "LP06 replaces FULLRES (6): "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU --extra_runs $SIX 2>&1 | grep MEAN
echo -n "LP06 added as 7th        : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_FULLRES $D $CU --extra_runs $SIX,$OUT 2>&1 | grep MEAN
echo "LP06 PROBE DONE $(date -u +%H:%M)"
