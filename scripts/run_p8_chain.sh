#!/bin/bash
# Run the 5 P8 test-case trainings as its own sequential chain, gated on GPU headroom
# so it can proceed in parallel with the main queue without OOM.
Q=/home/intern_2603055/vvc/queue/p8_runs.txt
n=0
while read -r L; do
  [ -z "$L" ] && continue
  n=$((n+1))
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 34000 ]; do sleep 120; done
  echo "== P8 chain job $n START $(date -u +%m-%d\ %H:%M)"
  bash -c "$L"
  echo "== P8 chain job $n END   $(date -u +%m-%d\ %H:%M)"
done < $Q
echo "P8 CHAIN DONE $(date)"
