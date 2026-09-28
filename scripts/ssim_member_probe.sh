#!/bin/bash
# SSIM-specialist ensemble member: full-res (so SSIM is optimised at the evaluated 4K scale) with a higher
# SSIM loss weight. Goal: raise ensemble SSIM (both FULL and FG) = the budget we convert to LPIPS via Difix.
C=001_1_seq0; R=/home/intern_2603055/vvc/runs/siga_$C; D=/home/intern_2603055/vvc/data/$C
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
OUT=$R/run_FULLRES_SSIM05
if [ ! -f $OUT/gaussians.pt ]; then
  echo "== train FULLRES_SSIM05 (data.scale=1.0, l1 0.5 / ssim 0.5, LPIPS_W 0.3) $(date -u +%H:%M)"
  FTGSPP_L1_W=0.5 FTGSPP_SSIM_W=0.5 FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_CHECKPOINT_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 data.scale=1.0 \
    train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_${C}_FULLRES_SSIM05.log 2>&1 || { echo FAILED; exit 1; }
fi
SIX="$R/run_P8,$R/run_P9,$R/run_J_dense01,$R/run_GATEONLY,$R/run_G4M"
echo "ref 6-member (FULLRES+5): 26.528/0.9348/0.2250"
echo -n "SSIM05 solo                : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU 2>&1 | grep MEAN
echo -n "SSIM05 replaces FULLRES (6): "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU --extra_runs $SIX 2>&1 | grep MEAN
echo -n "SSIM05 added as 7th        : "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_FULLRES $D $CU --extra_runs $SIX,$OUT 2>&1 | grep MEAN
echo "SSIM MEMBER PROBE DONE $(date -u +%H:%M)"
