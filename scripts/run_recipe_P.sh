#!/bin/bash
# Like run_recipe.sh but with per-case pseudo-view supervision (seva). Usage: run_recipe_P.sh <name> "<hydra overrides>" [ENV=VAL ...]
# Requires work/seva/pseudo_<prefix>/index.json for each case (built by export_pseudo.py).
set -u
NAME=$1; OVR=$2; shift 2
for kv in "$@"; do export "$kv"; done
export FTGSPP_TRAIN_EVAL_INTERVAL=0
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
for c in $CASES; do
  p=${c%_seq0}; OUT=/home/intern_2603055/vvc/runs/siga_$c/run_$NAME; PD=/home/intern_2603055/vvc/work/seva/pseudo_$p
  if [ ! -f $OUT/gaussians.pt ]; then
    if [ ! -f $PD/index.json ]; then echo "!! no pseudo views for $c ($PD)"; continue; fi
    echo "== train $c $NAME $(date)"
    FTGSPP_PSEUDO_DIR=$PD FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$c FTGSPP_POINTS_DIR=/home/intern_2603055/vvc/runs/siga_$c/points_stride10 \
      /home/intern_2603055/vvc/ftg.sh -m ftgspp.run data=siga_$c run.gpu=0 $OVR run.output_path=$OUT > /home/intern_2603055/vvc/runs_${c}_$NAME.log 2>&1
  fi
  if [ -f $OUT/gaussians.pt ]; then
    echo "== render $c $NAME $(date)"
    /home/intern_2603055/vvc/ftg.sh /home/intern_2603055/vvc/work/render_hidden.py $OUT /home/intern_2603055/vvc/data/$c /home/intern_2603055/vvc/submissions/$NAME/renders/$c --cull_near_frac 1.1 --cull_min_views 4 > /home/intern_2603055/vvc/render_${c}_$NAME.log 2>&1
  else echo "!! $c $NAME failed"; fi
done
python3 /home/intern_2603055/vvc/work/make_submission.py Doreen071 /home/intern_2603055/vvc/submissions/$NAME/renders /home/intern_2603055/vvc/submissions/${NAME}_submission.zip
echo "RECIPE $NAME DONE $(date)"
