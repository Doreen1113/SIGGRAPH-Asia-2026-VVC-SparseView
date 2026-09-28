#!/bin/bash
# Corrected approach: enable marginal_gating so static content uses the gate (not duration),
# freeing duration regularization to target only genuinely-dynamic (person) gaussians.
C=001_1_seq0; R=/home/intern_2603055/vvc/runs/siga_$C; D=/home/intern_2603055/vvc/data/$C
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
run() {
  local n=$1 gate=$2 w=$3
  local OUT=$R/run_$n
  if [ ! -f $OUT/gaussians.pt ]; then
    while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 30000 ]; do sleep 120; done
    echo "== train $n (gating=$gate reg_duration=$w) $(date -u +%H:%M)"
    FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 FTGSPP_REG_DURATION=$w \
    FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
      /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 model.marginal_gating=$gate \
      train.color_correction=true train.lpips_loss=true run.output_path=$OUT \
      > /home/intern_2603055/vvc/runs_${C}_$n.log 2>&1 || { echo "$n FAILED"; return; }
  fi
  echo -n "$n solo    : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU 2>&1 | grep MEAN
  echo -n "$n +P8P9J  : "; /home/intern_2603055/vvc/ftg.sh $EV $OUT $D $CU --extra_runs $R/run_P8,$R/run_P9,$R/run_J_dense01 2>&1 | grep MEAN
  echo -n "$n view18 FG-PSNR : "; FGDIAG_MAIN=run_$n /home/intern_2603055/vvc/ftg.sh /tmp/claude-1001/fg_diagnose.py $C /tmp/claude-1001/fgdiag_$n 2>&1 | grep "^18:"
}
echo "baseline S03 (gating off) solo 25.517/0.9205/0.2356 ; +P8P9J 26.107/0.9336/0.2256 ; view18 FG-PSNR=17.49"
run GATEONLY  true 0.0     # control: does gating alone (no duration reg) change anything?
run GATEDUR005 true 0.005  # gating + light duration reg on the (1-gate) subset
run GATEDUR02  true 0.02   # gating + stronger duration reg
echo "REGDUR V2 DONE $(date -u +%H:%M)"
