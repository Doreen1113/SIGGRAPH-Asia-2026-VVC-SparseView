#!/bin/bash
#SBATCH --job-name=bgmean_pack
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/bgmean_pack_%j.log
# Deploy an output-level background average validated by val_bgv2: build the mean tree of existing projected
# backgrounds, then swap it into the packed HB3_p085_six.zip (exact identity, swap_bg.py).
#   sbatch --export=ALL,NAME=m1510,MIX="NBX15_pAV:1 NBX10_pAV:1" work/slurm/bgmean_pack.sbatch
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${NAME:?}" "${MIX:?}"
BG=$V/submissions/BG_$NAME
ARGS=""; for m in $MIX; do ARGS="$ARGS $V/submissions/${m%%:*}/renders:${m##*:}"; done
echo "== mean tree $NAME <- $MIX, $(date +%H:%M) =="
$PY $W/weighted_ensemble.py $BG/renders $ARGS --procs 90
n=$(find $BG/renders -name '*.jpg' | wc -l); echo "BG_$NAME: $n/2056"; [ "$n" -eq 2056 ] || { echo MEAN_TREE_INCOMPLETE; exit 1; }
OUT=$V/submissions/HB3_p085_$NAME
echo "== swap_bg, $(date +%H:%M) =="
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY $W/swap_bg.py $V/submissions/HB3_p085_six.zip $V/submissions/SIX_pAVt15/renders $BG/renders $OUT/renders --shard $g/8 > $V/jobs/swapbg_${NAME}_$g.log 2>&1 &
done; wait
n=$(find $OUT/renders -name '*.jpg' | wc -l); echo "-- HB3_p085_$NAME: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/HB3_p085_$NAME.zip $OUT/renders && chmod 444 $V/submissions/HB3_p085_$NAME.zip
echo BGMEAN_PACK_DONE $(date +%H:%M)
