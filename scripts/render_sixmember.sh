#!/bin/bash
# Render the 6-member ensemble (FULLRES+P8+P9+J+GATE+G4M) on all 5 test cases.
W=/home/intern_2603055/vvc/work; S=/home/intern_2603055/vvc/submissions
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  R=/home/intern_2603055/vvc/runs/siga_$c
  OUT=$S/SIX_ens/renders/$c
  [ -d $OUT ] && continue
  echo "== render SIX $c $(date -u +%H:%M)"
  /home/intern_2603055/vvc/ftg.sh $W/render_hidden.py $R/run_FULLRES_s0 /home/intern_2603055/vvc/data/$c $OUT \
    --cull_near_frac 1.1 --cull_min_views 4 \
    --extra_runs $R/run_P8_s0,$R/run_P9_s0,$R/run_J_s0,$R/run_GATE_s0,$R/run_G4M_s0 \
    > /home/intern_2603055/vvc/render_${c}_SIX.log 2>&1 || { echo "$c RENDER FAILED"; rm -rf $OUT; }
done
echo "SIX-MEMBER RENDER DONE $(date -u +%H:%M)"
