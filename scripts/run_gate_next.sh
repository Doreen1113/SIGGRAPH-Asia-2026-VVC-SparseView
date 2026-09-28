#!/bin/bash
# (a) two-scene gate for marginal_gating on val 012_0   (b) does gating stack with the G4M capacity gain?
EV=/home/intern_2603055/vvc/work/eval_full.py; CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
train() { # name scene extra-overrides...
  local n=$1 C=$2; shift 2
  local R=/home/intern_2603055/vvc/runs/siga_$C; local OUT=$R/run_$n
  [ -f $OUT/gaussians.pt ] && return
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 30000 ]; do sleep 120; done
  echo "== train $n on $C $(date -u +%H:%M) [$*]"
  FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$C FTGSPP_POINTS_DIR=$R/points_stride10 FTGSPP_TRAIN_EVAL_INTERVAL=0 \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$C run.gpu=0 \
    train.color_correction=true train.lpips_loss=true "$@" run.output_path=$OUT \
    > /home/intern_2603055/vvc/runs_${C}_$n.log 2>&1 || echo "$n FAILED"
}
# (a) gate on the second val scene
train GATEONLY 012_0_seq0 model.marginal_gating=true
R2=/home/intern_2603055/vvc/runs/siga_012_0_seq0; D2=/home/intern_2603055/vvc/data/012_0_seq0
echo "012_0 baseline S03+P8+P9+J = 24.831/0.9193/0.2688"
echo -n "012_0 GATEONLY+P8P9J : "; /home/intern_2603055/vvc/ftg.sh $EV $R2/run_GATEONLY $D2 $CU --extra_runs $R2/run_P8,$R2/run_P9,$R2/run_J 2>&1 | grep MEAN
# (b) does gating stack with capacity?
train GATE_G4M 001_1_seq0 model.marginal_gating=true init.num_gaussians=4000000
R1=/home/intern_2603055/vvc/runs/siga_001_1_seq0; D1=/home/intern_2603055/vvc/data/001_1_seq0
echo "001_1 refs: base 26.107 | GATEONLY 26.344 | G4M 26.373"
echo -n "001_1 GATE_G4M solo   : "; /home/intern_2603055/vvc/ftg.sh $EV $R1/run_GATE_G4M $D1 $CU 2>&1 | grep MEAN
echo -n "001_1 GATE_G4M +P8P9J : "; /home/intern_2603055/vvc/ftg.sh $EV $R1/run_GATE_G4M $D1 $CU --extra_runs $R1/run_P8,$R1/run_P9,$R1/run_J_dense01 2>&1 | grep MEAN
echo "GATE NEXT DONE $(date -u +%H:%M)"
