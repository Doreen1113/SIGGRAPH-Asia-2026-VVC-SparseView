#!/bin/bash
# Ensemble-render hidden views for all 5 test cases from several recipe runs, then zip.
# Usage: final_render.sh <submission_name> <run_name1> [run_name2 ...]   e.g. final_render.sh F_ens3 F_s0 F_s1 F_s2
set -u
NAME=$1; shift; RUNS=("$@")
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
for c in $CASES; do
  MAIN=/home/intern_2603055/vvc/runs/siga_$c/run_${RUNS[0]}
  EXTRA=""
  for r in "${RUNS[@]:1}"; do d=/home/intern_2603055/vvc/runs/siga_$c/run_$r; [ -f $d/gaussians.pt ] && EXTRA="$EXTRA,$d"; done
  EXTRA=${EXTRA#,}
  if [ ! -f $MAIN/gaussians.pt ]; then echo "!! missing $MAIN"; continue; fi
  echo "== render $c main=$MAIN extra=$EXTRA $(date)"
  /home/intern_2603055/vvc/ftg.sh /home/intern_2603055/vvc/work/render_hidden.py $MAIN /home/intern_2603055/vvc/data/$c \
     /home/intern_2603055/vvc/submissions/$NAME/renders/$c --cull_near_frac 1.1 --cull_min_views 4 ${EXTRA:+--extra_runs $EXTRA} $( [ -d /home/intern_2603055/vvc/depth_vggt/$c ] && [ $(ls /home/intern_2603055/vvc/depth_vggt/$c | wc -l) -gt 50 ] && echo --vggt_depth_dir /home/intern_2603055/vvc/depth_vggt/$c ) 2>&1 | grep -v Warn | tail -1
done
python3 /home/intern_2603055/vvc/work/make_submission.py Doreen071 /home/intern_2603055/vvc/submissions/$NAME/renders /home/intern_2603055/vvc/submissions/${NAME}_submission.zip
echo "FINAL $NAME DONE $(date)"
