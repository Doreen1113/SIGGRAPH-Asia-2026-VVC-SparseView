#!/bin/bash
export HF_HOME=/home/intern_2603055/.cache/huggingface
for ts in 100 50; do for s in 001_1 012_0; do
  echo "== t=$ts $s $(date -u +%H:%M)"; /home/intern_2603055/vvc/envs/difix/bin/python /home/intern_2603055/vvc/work/difix_run.py /home/intern_2603055/vvc/fixtest/$s $ts difix_t$ts 2>&1 | grep -v arn | tail -1
  /home/intern_2603055/vvc/ftg.sh /home/intern_2603055/vvc/work/fix_metrics.py /home/intern_2603055/vvc/fixtest/$s render difix difix_t$ts 2>&1 | grep -E "^render|^difix"
done; done; echo "TSWEEP DONE $(date -u +%H:%M)"
