#!/bin/bash
# Train a recipe on the 5 sparse test cases, render hidden views, package submission.
# Usage: run_recipe.sh <recipe_name> "<hydra overrides>" [env assignments as VAR=VAL ...]
# Example: run_recipe.sh cc_lpips "train.color_correction=true train.lpips_loss=true" FTGSPP_DEPTH_W=0.2
set -u
NAME=$1; OVR=$2; shift 2
for kv in "$@"; do export "$kv"; done
export FTGSPP_TRAIN_EVAL_INTERVAL=0
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
for c in $CASES; do
  OUT=/home/intern_2603055/vvc/runs/siga_$c/run_$NAME
  if [ ! -f $OUT/gaussians.pt ]; then
    echo "== train $c $NAME $(date)"
    PDIR=/home/intern_2603055/vvc/runs/siga_$c/points_stride10; EXTRA_OVR=""
    if [ "${RECIPE_VGGT:-0}" = "1" ]; then
      PDIR=/home/intern_2603055/vvc/runs/siga_$c/points_vggt10; EXTRA_OVR="init.points_path=$PDIR"
      export FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$c
      [ -d $PDIR ] || ~/miniconda3/envs/gsplat_src/bin/python /home/intern_2603055/vvc/work/vggt_points.py /home/intern_2603055/vvc/data/$c /home/intern_2603055/vvc/depth_vggt/$c $PDIR > /home/intern_2603055/vvc/points_vggt_$c.log 2>&1
    elif [ -n "${FTGSPP_DENSE_W:-}" ]; then
      export FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$c
    fi
    FTGSPP_POINTS_DIR=$PDIR \
      /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$c run.gpu=0 $OVR $EXTRA_OVR run.output_path=$OUT > /home/intern_2603055/vvc/runs_${c}_$NAME.log 2>&1
  fi
  if [ -f $OUT/gaussians.pt ]; then
    echo "== render $c $NAME $(date)"
    /home/intern_2603055/vvc/ftg.sh /home/intern_2603055/vvc/work/render_hidden.py $OUT /home/intern_2603055/vvc/data/$c /home/intern_2603055/vvc/submissions/$NAME/renders/$c --cull_near_frac 1.1 --cull_min_views 4 > /home/intern_2603055/vvc/render_${c}_$NAME.log 2>&1
  else
    echo "!! $c $NAME failed"; fi
done
python3 /home/intern_2603055/vvc/work/make_submission.py Doreen071 /home/intern_2603055/vvc/submissions/$NAME/renders /home/intern_2603055/vvc/submissions/${NAME}_submission.zip
echo "RECIPE $NAME DONE $(date)"
