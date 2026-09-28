#!/bin/bash
# 012_0 gate check for perceptual projection: build the repo199 Difix baseline there, then run the best-so-far
# lambda values and compare against the dense uniform-blend curve on THIS scene (never assume a 001_1 result
# generalises without checking).
cd /home/intern_2603055/projects/Difix3D; export HF_HOME=/home/intern_2603055/.cache/huggingface
until grep -q "PROJECTION2 DONE\|FAILED" /home/intern_2603055/vvc/projection2.log 2>/dev/null; do sleep 90; done
/home/intern_2603055/vvc/envs/difix/bin/python /home/intern_2603055/vvc/work/difix_ft_apply.py /home/intern_2603055/vvc/fixtest/012_0 repo199 --blend 0.30,0.40,0.50,0.65 > /home/intern_2603055/vvc/gate012_apply.log 2>&1 || { echo FAILED-apply; exit 1; }
cd /home/intern_2603055/vvc
python3 work/blend_sweep.py fixtest/012_0 repo199 0.30,0.40,0.50,0.65 > /dev/null
for L in 10 15 20; do
  ./ftg.sh /home/intern_2603055/vvc/work/perceptual_projection.py fixtest/012_0 repo199 repo199a050 $L 200 proj_l$L > /home/intern_2603055/vvc/gate012_proj_l$L.log 2>&1 || echo "FAILED l$L"
done
echo "== 012_0 gate: dense uniform curve vs projection =="
./ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py /home/intern_2603055/vvc/fixtest/012_0 render repo199a030 repo199a040 repo199a050 repo199a065 proj_l10 proj_l15 proj_l20 2>&1 | grep -E "^(render|repo|proj)" | cut -c1-72
echo "GATE012 DONE $(date -u +%H:%M)"
