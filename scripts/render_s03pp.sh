#!/bin/bash
# Final recipe: S03 + P8 + P9 + J_s0, averaged, near-camera cull. No sharpening (costs our SSIM lead).
W=/home/intern_2603055/vvc/work; S=/home/intern_2603055/vvc/submissions; RN=/home/intern_2603055/vvc/runs
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  R=$RN/siga_$c
  [ -d $S/S03PPJ_ens/renders/$c ] && continue
  echo "== render S03PPJ $c $(date -u +%m-%d\ %H:%M)"
  /home/intern_2603055/vvc/ftg.sh $W/render_hidden.py $R/run_S03_s0 /home/intern_2603055/vvc/data/$c \
    $S/S03PPJ_ens/renders/$c --cull_near_frac 1.1 --cull_min_views 4 \
    --extra_runs $R/run_P8_s0,$R/run_P9_s0,$R/run_J_s0 \
    > /home/intern_2603055/vvc/render_${c}_S03PPJ.log 2>&1 || rm -rf $S/S03PPJ_ens/renders/$c
done
echo "S03PPJ RENDER DONE $(date)"
