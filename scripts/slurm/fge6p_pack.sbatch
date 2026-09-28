#!/bin/bash
#SBATCH --job-name=fge6p_pack
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:55:00
#SBATCH --output=/work/doreen071/vvc/jobs/fge6p_pack_%j.log
# Person = projected FG ensemble (alpha 0.85), background = the unchanged SIX_pAVt15. Two alphas so we have a
# hedge ready without another 4h window.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ "$(find $V/submissions/FGE6_pAVt15 -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ]; do sleep 120; done
i=0
for PA in 0.85 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/FGE6_ens/renders $V/submissions/FGE6_pAVt15/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/FGE6P_${TAG}_bpAVt15/renders --cases $c > $V/jobs/fge6pcomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 085 070; do
  OUT=FGE6P_p${PA}_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
  echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo FGE6P_PACK_DONE $(date +%H:%M)
