#!/bin/bash
cd /home/intern_2603055/vvc
PAIRS="0.50:0.45,0.55:0.50,0.60:0.50,0.60:0.55,0.65:0.55,0.70:0.60,0.50:0.55,0.55:0.55"
for D in fixtest/001_1 fixtest/012_0; do
  ./ftg.sh /home/intern_2603055/vvc/work/person_bg_sweep.py /home/intern_2603055/vvc/$D repo199tta "$PAIRS" > /home/intern_2603055/vvc/pbsweep_$(basename $D).log 2>&1 || echo "FAILED $D"
done
echo "== TTA person/bg alpha sweep, both val scenes =="
for D in fixtest/001_1 fixtest/012_0; do
  echo "-- $D --"
  V="render"; for p in 050_045 055_050 060_050 060_055 065_055 070_060 050_055 055_055; do V="$V pb_p$p"; done
  ./ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py /home/intern_2603055/vvc/$D $V 2>&1 | grep -E "^(render|pb)" | cut -c1-90
done
echo "PBSWEEP DONE $(date -u +%H:%M)"
