#!/bin/bash
#SBATCH --job-name=bgv3_pack
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/bgv3_pack_%j.log
# Waits for the full projected background, then swaps it into the live best zip (exact identity) and packs.
#   sbatch --export=ALL,MIX=ownT,LAM=15 work/slurm/bgv3_pack.sbatch
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${MIX:=ownT}" "${LAM:=15}"
BG=$V/submissions/BGV3_${MIX}_l${LAM}_pAV/renders; NAME=HB3_p085_bgv3_${MIX}_l${LAM}; OUT=$V/submissions/$NAME
until [ "$(find $BG -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ]; do sleep 60; done
echo "== swap_bg NBX15 -> $MIX l$LAM, $(date +%H:%M) =="
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY $W/swap_bg.py $V/submissions/HB3_p085_nbx15s.zip $V/submissions/NBX15_pAV/renders $BG $OUT/renders --shard $g/8 > $V/jobs/swapbg_bgv3_$g.log 2>&1 &
done; wait
n=$(find $OUT/renders -name '*.jpg' | wc -l); echo "-- $NAME: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$NAME.zip $OUT/renders && chmod 444 $V/submissions/$NAME.zip
echo BGV3_PACK_DONE $(date +%H:%M)
