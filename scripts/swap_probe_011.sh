#!/bin/bash
# Render the proxy-validated 5-member composition for 011_0, repackage, and swap it into the armed slot.
W=/home/intern_2603055/vvc/work; S=/home/intern_2603055/vvc/submissions; R=/home/intern_2603055/vvc/runs/siga_011_0_seq0
OUT=$S/SC_5S015/renders/011_0_seq0
if [ ! -d $OUT ]; then
  echo "== render 011_0 S03+S015+P8+P9+J_s0 $(date -u +%H:%M)"
  /home/intern_2603055/vvc/ftg.sh $W/render_hidden.py $R/run_S03_s0 /home/intern_2603055/vvc/data/011_0_seq0 $OUT \
    --cull_near_frac 1.1 --cull_min_views 4 --extra_runs $R/run_S015_s0,$R/run_P8_s0,$R/run_P9_s0,$R/run_J_s0 \
    > /home/intern_2603055/vvc/render_011_0_SC_5S015.log 2>&1 || { rm -rf $OUT; echo "RENDER FAILED - keeping ALL5 probe"; exit 1; }
fi
n=$(find $OUT -name '*.jpg' | wc -l); [ "$n" = "360" ] || { echo "INTEGRITY FAIL ($n) - keeping ALL5 probe"; exit 1; }
rm -rf /tmp/claude-1001/next2; mkdir -p /tmp/claude-1001/next2/renders
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0; do cp -r $S/S03PPJ_ens/renders/$c /tmp/claude-1001/next2/renders/; done
cp -r $OUT /tmp/claude-1001/next2/renders/
python3 $W/mix_submission.py /home/intern_2603055/vvc/probe_011_5s015.zip /tmp/claude-1001/next2/renders | tail -1
for p in $(ps -eo pid,args | grep -v grep | grep "submit_when_ready.sh" | awk '{print $1}'); do kill $p; done; sleep 2
setsid nohup $W/submit_when_ready.sh /home/intern_2603055/vvc/probe_011_5s015.zip probe_011_5s015.zip > /home/intern_2603055/vvc/submit_probe1.log 2>&1 < /dev/null &
echo "SWAPPED to 5S015 probe $(date -u +%H:%M)"
