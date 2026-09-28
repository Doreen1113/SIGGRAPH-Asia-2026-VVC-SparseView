#!/bin/bash
# Evaluate fine-tuned Difix checkpoints on val 001_1: early / mid / final, each at several raw-blend alphas.
# Key number to extract: LPIPS gained per 0.001 SSIM lost, vs the generic Difix's measured 0.00315.
cd /home/intern_2603055/projects/Difix3D
export HF_HOME=/home/intern_2603055/.cache/huggingface
CK=/home/intern_2603055/vvc/difix_ft/ckpt/checkpoints
PY=/home/intern_2603055/vvc/envs/difix/bin/python
APPLY=/home/intern_2603055/vvc/work/difix_ft_apply.py
DUMP=/home/intern_2603055/vvc/fixtest/001_1
V="render difix repo199 repo199b45"
for N in 1751 1001 501; do
  echo "== apply ft$N $(date -u +%H:%M)"
  $PY $APPLY $DUMP ft$N --ckpt $CK/model_$N.pkl --blend 0.3,0.45,0.6 > /home/intern_2603055/vvc/ft_apply_001_1_$N.log 2>&1 \
    && V="$V ft$N ft${N}b30 ft${N}b45 ft${N}b60" || echo "FAILED ft$N"
  echo "== metrics so far $(date -u +%H:%M)"
  /home/intern_2603055/vvc/ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py $DUMP $V 2>&1 | grep -E "^(render|difix|repo|ft)"
done
echo "FT EVAL DONE $(date -u +%H:%M)"
