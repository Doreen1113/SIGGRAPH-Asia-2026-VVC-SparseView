#!/bin/bash
# Combine the two winning techniques: perceptual projection targeting the TTA (3-shift) Difix output instead of
# single-shift. If gains stack, this should beat both individually.
cd /home/intern_2603055/vvc; DUMP=/home/intern_2603055/vvc/fixtest/001_1
for L in 15 20 30 45; do
  ./ftg.sh /home/intern_2603055/vvc/work/perceptual_projection.py $DUMP repo199tta repo199a050 $L 200 projtta_l$L > /home/intern_2603055/vvc/projtta_l$L.log 2>&1 || echo "FAILED l$L"
done
echo "== projection targeting TTA output, val 001_1 =="
./ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py $DUMP render repo199a040 repo199a050 proj_l15 proj_l20 proj_l30 projtta_l15 projtta_l20 projtta_l30 projtta_l45 2>&1 | grep -E "^(render|repo|proj)" | cut -c1-90
echo "PROJTTA DONE $(date -u +%H:%M)"
