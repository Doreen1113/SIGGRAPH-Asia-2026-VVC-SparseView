#!/bin/bash
cd /home/intern_2603055/vvc; DUMP=/home/intern_2603055/vvc/fixtest/001_1
for L in 3 10; do
  ./ftg.sh /home/intern_2603055/vvc/work/perceptual_projection.py $DUMP repo199 repo199a050 $L 200 proj_l$L > /home/intern_2603055/vvc/proj_l$L.log 2>&1 || echo "FAILED l$L"
done
echo "== projection (init = uniform a050) =="
./ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py $DUMP render repo199a030 repo199a050 repo199a065 proj_l3 proj_l10 2>&1 | grep -E "^(render|repo|proj)" | cut -c1-72
echo "PROJECTION DONE $(date -u +%H:%M)"
