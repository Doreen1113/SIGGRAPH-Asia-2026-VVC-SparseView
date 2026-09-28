#!/bin/bash
# After the fine-tune finishes, apply each checkpoint to both val dumps at several blend alphas and score them.
# What we are looking for: a better SSIM->LPIPS exchange rate than the generic Difix's measured 0.00315 per 0.001.
while pgrep -f train_difix.py > /dev/null; do sleep 120; done
echo "fine-tune finished $(date -u +%H:%M); GPU free, starting checkpoint evaluation"
cd /home/intern_2603055/projects/Difix3D
export HF_HOME=/home/intern_2603055/.cache/huggingface
CK=/home/intern_2603055/vvc/difix_ft/ckpt/checkpoints
PY=/home/intern_2603055/vvc/envs/difix/bin/python
APPLY=/home/intern_2603055/vvc/work/difix_ft_apply.py
for D in 001_1 012_0; do
  DUMP=/home/intern_2603055/vvc/fixtest/$D
  for N in 501 1001 1501; do
    [ -f $CK/model_$N.pkl ] || continue
    echo "== apply ft$N to $D $(date -u +%H:%M)"
    $PY $APPLY $DUMP ft$N --ckpt $CK/model_$N.pkl --blend 0.3,0.45,0.6 > /home/intern_2603055/vvc/ft_apply_${D}_$N.log 2>&1 || echo "FAILED ft$N $D"
  done
  echo "== metrics for $D =="
  V="render difix"
  for N in 501 1001 1501; do [ -f $DUMP/$(python3 -c "import json;print(json.load(open('$DUMP/manifest.json'))[0]['tag'])")_ft$N.png ] && V="$V ft$N ft${N}b30 ft${N}b45 ft${N}b60"; done
  /home/intern_2603055/vvc/ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py $DUMP $V 2>&1 | grep -E "^(render|difix|ft)"
done
echo "FT CKPT EVAL DONE $(date -u +%H:%M)"
