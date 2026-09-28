#!/bin/bash
cd /home/intern_2603055/vvc
python3 work/blend_sweep.py fixtest/001_1 repo199 0.45,0.60 > /dev/null
python3 work/adaptive_blend.py fixtest/001_1 repo199 0.30,0.45,0.60 15 > /dev/null
echo "== adaptive (rad 15) vs uniform at equal mean alpha =="
./ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py /home/intern_2603055/vvc/fixtest/001_1 render repo199a030 repo199adp030 repo199a045 repo199adp045 repo199a060 repo199adp060 2>&1 | grep -E "^(render|repo)" | cut -c1-72
echo "ADAPTIVE DONE $(date -u +%H:%M)"
