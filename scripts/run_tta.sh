#!/bin/bash
# Difix self-ensemble via shifted tile grids: averages out grid-dependent hallucination, keeps consistent corrections.
cd /home/intern_2603055/projects/Difix3D; export HF_HOME=/home/intern_2603055/.cache/huggingface
PY=/home/intern_2603055/vvc/envs/difix/bin/python; APPLY=/home/intern_2603055/vvc/work/difix_ft_apply.py; DUMP=/home/intern_2603055/vvc/fixtest/001_1
$PY $APPLY $DUMP repo199s1 --shift 320,288 > /home/intern_2603055/vvc/tta_s1.log 2>&1 || echo "FAILED s1"
$PY $APPLY $DUMP repo199s2 --shift 640,192 > /home/intern_2603055/vvc/tta_s2.log 2>&1 || echo "FAILED s2"
cd /home/intern_2603055/vvc
python3 - <<'PYX'
import json; from pathlib import Path; import numpy as np; from PIL import Image
d=Path('/home/intern_2603055/vvc/fixtest/001_1')
for e in json.load(open(d/'manifest.json')):
    ims=[np.asarray(Image.open(d/f"{e['tag']}_{v}.png").convert('RGB'),np.float32) for v in ('repo199','repo199s1','repo199s2')]
    Image.fromarray(np.clip(np.mean(ims,0),0,255).astype(np.uint8)).save(d/f"{e['tag']}_repo199tta.png")
PYX
python3 work/blend_sweep.py fixtest/001_1 repo199tta 0.30,0.45,0.60,0.80,1.00 > /dev/null
echo "== TTA (3 shifted grids averaged) vs single, along the blend curve =="
./ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py $DUMP render repo199a030 repo199ttaa030 repo199a045 repo199ttaa045 repo199a060 repo199ttaa060 repo199a080 repo199ttaa080 repo199a100 repo199ttaa100 2>&1 | grep -E "^(render|repo)" | cut -c1-72
echo "TTA DONE $(date -u +%H:%M)"
