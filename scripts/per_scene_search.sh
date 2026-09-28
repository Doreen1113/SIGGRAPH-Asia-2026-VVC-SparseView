#!/bin/bash
# 006_1 and 011_0 swing 0.9-1.3 dB across recipes while the other three scenes move <0.4.
# Render several ensemble compositions for just these two scenes so we can pick per-scene from real feedback.
W=/home/intern_2603055/vvc/work; S=/home/intern_2603055/vvc/submissions; RN=/home/intern_2603055/vvc/runs
render() { # name case extras...
  local name=$1 c=$2 main=$3 extras=$4
  [ -d $S/$name/renders/$c ] && return
  echo "== render $name $c $(date -u +%m-%d\ %H:%M)"
  /home/intern_2603055/vvc/ftg.sh $W/render_hidden.py $RN/siga_$c/$main /home/intern_2603055/vvc/data/$c \
    $S/$name/renders/$c --cull_near_frac 1.1 --cull_min_views 4 --extra_runs "$extras" \
    > /home/intern_2603055/vvc/render_${c}_${name}.log 2>&1 || rm -rf $S/$name/renders/$c
}
for c in 006_1_seq0 011_0_seq0; do
  R=$RN/siga_$c
  # A: S03 primary with J_s0 + J_s1 (two J seeds) instead of P8/P9
  render SC_S03JJ $c run_S03_s0 "$R/run_J_s0,$R/run_J_s1"
  # B: J_s0 primary + P8 + P9 (the J_ens-flavoured mix that 006_1 liked, plus pseudo members)
  render SC_JPP $c run_J_s0 "$R/run_P8_s0,$R/run_P9_s0"
  # C: five members
  render SC_ALL5 $c run_S03_s0 "$R/run_P8_s0,$R/run_P9_s0,$R/run_J_s0,$R/run_J_s1"
done
echo "PER-SCENE SEARCH DONE $(date)"
