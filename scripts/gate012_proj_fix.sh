#!/bin/bash
cd /home/intern_2603055/vvc; DUMP=/home/intern_2603055/vvc/fixtest/012_0
for L in 10 15 20; do
  ./ftg.sh /home/intern_2603055/vvc/work/perceptual_projection.py $DUMP repo199 repo199a050 $L 200 proj_l$L > /home/intern_2603055/vvc/gate012_proj_l$L.log 2>&1 || echo "FAILED l$L"
done
echo "== 012_0 gate: dense uniform curve vs projection =="
./ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py $DUMP render repo199a030 repo199a040 repo199a050 repo199a065 proj_l10 proj_l15 proj_l20 2>&1 | grep -E "^(render|repo|proj)" | cut -c1-72
echo "GATE012 DONE $(date -u +%H:%M)"
