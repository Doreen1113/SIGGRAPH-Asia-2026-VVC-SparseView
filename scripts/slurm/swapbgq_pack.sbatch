#!/bin/bash
#SBATCH --job-name=swapbgq_pack
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/swapbgq_pack_%j.log
# The HB3 person source trees were removed in the quota emergency clean, so rebuild "HB3 person + cross-target
# background" by swapping the background of the already-packed HB3_*_six zips (exact identity, see swap_bg.py).
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for TAG in p085; do
  OUT=$V/submissions/HB3_${TAG}_nbx15q
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY $W/swap_bg.py $V/submissions/HB3_${TAG}_six.zip $V/submissions/SIX_pAVt15/renders \
      $V/submissions/NBX15_pAV/renders $OUT/renders --shard $g/8 --q 100 > $V/jobs/swapbg_${TAG}_$g.log 2>&1 &
  done
  wait
  n=$(find $OUT/renders -name '*.jpg' 2>/dev/null | wc -l); echo "-- HB3_${TAG}_nbx15q: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/HB3_${TAG}_nbx15q.zip $OUT/renders && chmod 444 $V/submissions/HB3_${TAG}_nbx15q.zip
done
echo SWAPBGQ_DONE $(date +%H:%M)
