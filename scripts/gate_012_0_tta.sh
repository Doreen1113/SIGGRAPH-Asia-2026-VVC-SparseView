#!/bin/bash
# 012_0 TTA gate check: build repo199 (already queued by gate_012_0.sh) plus two shifted variants, average, blend sweep.
cd /home/intern_2603055/projects/Difix3D; export HF_HOME=/home/intern_2603055/.cache/huggingface
PY=/home/intern_2603055/vvc/envs/difix/bin/python; APPLY=/home/intern_2603055/vvc/work/difix_ft_apply.py; DUMP=/home/intern_2603055/vvc/fixtest/012_0
until [ -f $DUMP/$(python3 -c "import json;print(json.load(open('$DUMP/manifest.json'))[0]['tag'])")_repo199.png ]; do sleep 60; done
$PY $APPLY $DUMP repo199s1 --shift 320,288 > /home/intern_2603055/vvc/gate012_tta_s1.log 2>&1 || echo FAILED-s1
$PY $APPLY $DUMP repo199s2 --shift 640,192 > /home/intern_2603055/vvc/gate012_tta_s2.log 2>&1 || echo FAILED-s2
cd /home/intern_2603055/vvc
python3 - <<'PYX'
import json; from pathlib import Path; import numpy as np; from PIL import Image
d=Path('/home/intern_2603055/vvc/fixtest/012_0')
for e in json.load(open(d/'manifest.json')):
    ims=[np.asarray(Image.open(d/f"{e['tag']}_{v}.png").convert('RGB'),np.float32) for v in ('repo199','repo199s1','repo199s2')]
    Image.fromarray(np.clip(np.mean(ims,0),0,255).astype(np.uint8)).save(d/f"{e['tag']}_repo199tta.png")
PYX
python3 work/blend_sweep.py fixtest/012_0 repo199tta 0.30,0.45,0.60,0.80,1.00 > /dev/null
echo "== 012_0 TTA gate =="
./ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py /home/intern_2603055/vvc/fixtest/012_0 render repo199a030 repo199ttaa030 repo199a045 repo199ttaa045 repo199a060 repo199ttaa060 repo199a080 repo199ttaa080 2>&1 | grep -E "^(render|repo)" | cut -c1-72
echo "GATE012TTA DONE $(date -u +%H:%M)"
