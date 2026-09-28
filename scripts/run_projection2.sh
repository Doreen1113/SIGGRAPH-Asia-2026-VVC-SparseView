#!/bin/bash
cd /home/intern_2603055/vvc
for L in 6 15 20 30; do
  ./ftg.sh /home/intern_2603055/vvc/work/perceptual_projection.py /home/intern_2603055/vvc/fixtest/001_1 repo199 repo199a050 $L 200 proj_l$L > /home/intern_2603055/vvc/proj_l$L.log 2>&1 || echo "FAILED l$L"
done
echo "== dense lambda sweep, val 001_1 =="
./ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py /home/intern_2603055/vvc/fixtest/001_1 render repo199a030 repo199a040 repo199a050 repo199a065 proj_l3 proj_l6 proj_l10 proj_l15 proj_l20 proj_l30 2>&1 | grep -E "^(render|repo|proj)" | cut -c1-72
echo "PROJECTION2 DONE $(date -u +%H:%M)"
