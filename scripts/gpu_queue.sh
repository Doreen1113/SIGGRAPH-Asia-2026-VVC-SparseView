#!/bin/bash
# Sequential GPU job queue. Each line of ~/vvc/queue/queue.txt is a shell command; runs when GPU mem < LIMIT MiB.
Q=/home/intern_2603055/vvc/queue/queue.txt; LIMIT=${GPU_QUEUE_LIMIT:-32000}
touch $Q
while true; do
  L=$(head -1 $Q)
  if [ -z "$L" ]; then sleep 120; continue; fi
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt $LIMIT ]; do sleep 120; done
  sed -i '1d' $Q
  echo "== START $(date -u +%m-%d\ %H:%M) :: $L"
  bash -c "$L"
  echo "== END   $(date -u +%m-%d\ %H:%M) :: $L"
done
