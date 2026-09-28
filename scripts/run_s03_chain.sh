#!/bin/bash
# S03 = J recipe at FTGSPP_LPIPS_W=0.3. At 4K in the ensemble it beats S015 on PSNR at equal LPIPS.
# 012_0 gate runs first, then the five test cases.
export FTGSPP_TRAIN_EVAL_INTERVAL=0
run_one() {
  local c=$1 out=$2 depth=$3 pts=$4 log=$5
  [ -f $out/gaussians.pt ] && return
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 34000 ]; do sleep 120; done
  echo "== train S03 $c $(date -u +%m-%d\ %H:%M)"
  FTGSPP_LPIPS_W=0.3 FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_DENSE_W=0.1 \
  FTGSPP_DENSE_DEPTH_DIR=$depth FTGSPP_POINTS_DIR=$pts \
    /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$c run.gpu=0 \
    train.color_correction=true train.lpips_loss=true run.output_path=$out > $log 2>&1
}
# two-scene gate first
run_one 012_0_seq0 /home/intern_2603055/vvc/runs/siga_012_0_seq0/run_S03 /home/intern_2603055/vvc/depth_vggt/012_0_seq0 /home/intern_2603055/vvc/runs/siga_012_0_seq0/points_stride10 /home/intern_2603055/vvc/runs_012_0_S03.log
echo "== 012_0 gate trained, evaluating @4K"
R=/home/intern_2603055/vvc/runs/siga_012_0_seq0; C=/home/intern_2603055/vvc/data/012_0_seq0; EV=/home/intern_2603055/vvc/work/eval_full.py
CU="--scale 1.0 --every 200 --cull_near_frac 1.1 --cull_min_views 4"
echo -n "012_0 S015+P8+J  +sharp: "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_S015 $C $CU --extra_runs $R/run_P8,$R/run_J --unsharp 0.4,4 2>&1 | grep MEAN
echo -n "012_0 S03 +P8+J  +sharp: "; /home/intern_2603055/vvc/ftg.sh $EV $R/run_S03  $C $CU --extra_runs $R/run_P8,$R/run_J --unsharp 0.4,4 2>&1 | grep MEAN
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  run_one $c /home/intern_2603055/vvc/runs/siga_$c/run_S03_s0 /home/intern_2603055/vvc/depth_vggt/$c /home/intern_2603055/vvc/runs/siga_$c/points_stride10 /home/intern_2603055/vvc/runs_${c}_S03_s0.log
done
echo "S03 CHAIN DONE $(date)"
