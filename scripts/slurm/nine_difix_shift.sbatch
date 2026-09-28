#!/bin/bash
#SBATCH --job-name=nine_difix
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/nine_difix_%x_%j.log
# One Difix shift over the 9-member ensemble, 8-way sharded. Resumable (skips existing outputs).
# Usage: sbatch --export=ALL,DX=320,DY=288,OUT=NINE_difix_s1 ...  (SLURM --export cannot carry commas in a value)
V=/work/doreen071/vvc; PY=/home/doreen071/.conda/envs/vvc/bin/python
SHIFT=""; [ -n "$DX" ] && SHIFT="$DX,$DY"
echo "shift=[$SHIFT] out=$OUT"
export OMP_NUM_THREADS=10
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $V/work/difix_submission_h200.py \
    $V/submissions/NINE_ens/renders $V/submissions/$OUT/renders ${SHIFT:+--shift $SHIFT} \
    > $V/jobs/nine_${OUT}_shard$i.log 2>&1 &
done
wait
echo "$OUT: $(find $V/submissions/$OUT -name '*.jpg' | wc -l)/2056"
echo NINE_DIFIX_${OUT}_DONE $(date -u +%H:%M)
