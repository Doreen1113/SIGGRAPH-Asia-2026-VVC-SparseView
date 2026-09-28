#!/bin/bash
# Train + render the 3 selfcap sparse cases (recipe J = F + dense VGGT depth). Waits for RoMa points from points_selfcap.log.
set -u
NAME=${1:-J}; USE_DENSE=${USE_DENSE:-0}
cd /home/intern_2603055/vvc
until grep -q "SELFCAP POINTS DONE" points_selfcap.log 2>/dev/null; do sleep 120; done
for c in 0512_bike 0525_corgi 0811_yoga; do
  OUT=/home/intern_2603055/vvc/runs/selfcap_$c/run_$NAME; PTS=/home/intern_2603055/vvc/runs/selfcap_$c/points_stride10
  if [ "$USE_DENSE" = "1" ] && { [ ! -d /home/intern_2603055/vvc/depth_vggt/$c ] || [ $(ls /home/intern_2603055/vvc/depth_vggt/$c 2>/dev/null | wc -l) -lt 900 ]; }; then
    echo "== vggt depth $c $(date)"
    PYTHONPATH=/home/intern_2603055/projects/vggt /home/intern_2603055/miniconda3/envs/vggt/bin/python work/vggt_depth.py data/$c /home/intern_2603055/vvc/depth_vggt/$c > /home/intern_2603055/vvc/vggt_$c.log 2>&1 || echo "!! vggt failed $c"
  fi
  DENSE=""; [ "$USE_DENSE" = "1" ] && [ $(ls /home/intern_2603055/vvc/depth_vggt/$c 2>/dev/null | wc -l) -ge 900 ] && DENSE="FTGSPP_DENSE_W=0.1 FTGSPP_DENSE_DEPTH_DIR=/home/intern_2603055/vvc/depth_vggt/$c"
  if [ ! -f $OUT/gaussians.pt ]; then
    echo "== train $c $NAME dense=[$DENSE] $(date)"
    env FTGSPP_DEPTH_W=0.2 FTGSPP_REG_OPACITY=0.05 FTGSPP_TRAIN_EVAL_INTERVAL=0 FTGSPP_POINTS_DIR=$PTS $DENSE \
      ./ftg.sh -m ftgspp.run data=selfcap_$c run.gpu=0 train.color_correction=true train.lpips_loss=true run.output_path=$OUT > /home/intern_2603055/vvc/runs_${c}_$NAME.log 2>&1
  fi
  if [ -f $OUT/gaussians.pt ]; then
    echo "== render $c $NAME $(date)"
    ./ftg.sh /home/intern_2603055/vvc/work/render_hidden.py $OUT /home/intern_2603055/vvc/data/$c /home/intern_2603055/vvc/submissions/selfcap/renders/$c --cull_near_frac 1.1 --cull_min_views 4 > /home/intern_2603055/vvc/render_${c}_$NAME.log 2>&1
  else echo "!! $c $NAME failed"; fi
done
ALL_OK=1
for c in 0512_bike 0525_corgi 0811_yoga; do [ -d /home/intern_2603055/vvc/submissions/selfcap/renders/$c ] || ALL_OK=0; done
[ "$ALL_OK" = "1" ] && touch /home/intern_2603055/vvc/submissions/selfcap/READY && echo "== touched selfcap/READY $(date)"
echo "SELFCAP RECIPE $NAME DONE $(date)"
