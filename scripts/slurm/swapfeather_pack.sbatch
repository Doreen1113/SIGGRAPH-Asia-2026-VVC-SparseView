#!/bin/bash
#SBATCH --job-name=swapft_pack
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/swapfeather_pack_%j.log
# Apply the feather widening (k25 -> K1) to the live best package HB3_p085_nbx15s.zip via swap_feather.py.
#   sbatch --export=ALL,K1=81,FB=B work/slurm/swapfeather_pack.sbatch
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${K1:?}" "${FB:=B}" "${SIG:=0}" "${T0:=0.15}" "${T1:=0.5}" "${TAG:=}"
FROOT=B; [ "$FB" != "B" ] && FROOT=$V/submissions/$FB/renders
NAME=HB3_p085_nbx15_f${K1}${TAG}; OUT=$V/submissions/$NAME
echo "== $NAME (fallback $FB), $(date +%H:%M) =="
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY $W/swap_feather.py tree $V/submissions/HB3_p085_nbx15s.zip $V/submissions/NBX15_pAV/renders $FROOT $OUT/renders \
    --k1 $K1 --sigma $SIG --t0 $T0 --t1 $T1 --shard $g/8 > $V/jobs/swapft_${K1}_$g.log 2>&1 &
done; wait
n=$(find $OUT/renders -name '*.jpg' | wc -l); echo "-- $NAME: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$NAME.zip $OUT/renders && chmod 444 $V/submissions/$NAME.zip
echo SWAPFT_PACK_DONE $(date +%H:%M)
