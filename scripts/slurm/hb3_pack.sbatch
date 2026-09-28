#!/bin/bash
#SBATCH --job-name=hb3_pack
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/hb3_pack_%j.log
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until (for d in HB3_difix HB3_difix_s1 HB3_difix_s2; do [ "$(find $V/submissions/$d -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ] || exit 1; done); do sleep 60; done
$PY $W/tta_average.py $V/submissions/HB3_TTA/renders $V/submissions/HB3_difix/renders $V/submissions/HB3_difix_s1/renders $V/submissions/HB3_difix_s2/renders --procs 90
i=0
for PA in 0.70 0.85; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/HB3_ens/renders $V/submissions/HB3_TTA/renders $PA \
      $V/submissions/NB_pAVt15/renders 1.0 $V/submissions/HB3_${TAG}_nbp/renders --cases $c > $V/jobs/hb3comp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for TAG in p070 p085; do
  OUT=HB3_${TAG}_nbp; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l); echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo HB3_PACK_DONE $(date +%H:%M)
