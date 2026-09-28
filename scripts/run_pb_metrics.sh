#!/bin/bash
cd /home/intern_2603055/vvc
echo "== TTA person/bg alpha sweep, both val scenes =="
for D in fixtest/001_1 fixtest/012_0; do
  echo "-- $D --"
  V="render"; for p in 050_b045 055_b050 060_b050 060_b055 065_b055 070_b060 050_b055 055_b055; do V="$V pb_p$p"; done
  ./ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py /home/intern_2603055/vvc/$D $V 2>&1 | grep -E "^(render|pb)" | cut -c1-90
done
echo "PBMETRICS DONE $(date -u +%H:%M)"
