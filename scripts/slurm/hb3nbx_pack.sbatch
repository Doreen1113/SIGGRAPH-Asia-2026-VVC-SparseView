#!/bin/bash
#SBATCH --job-name=hb3nbx_pack
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/hb3nbx_pack_%j.log
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for L in 15 10; do
  until [ "$(find $V/submissions/NBX${L}_pAV -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ]; do sleep 120; done
  i=0
  for PA in 0.70 0.85; do
    TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
    for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
      CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/HB3_ens/renders $V/submissions/HB3_TTA/renders $PA \
        $V/submissions/NBX${L}_pAV/renders 1.0 $V/submissions/HB3_${TAG}_nbx${L}/renders --cases $c > $V/jobs/hb3nbx_${TAG}_${L}_$c.log 2>&1 &
      i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
    done
  done
  wait
  for TAG in p070 p085; do
    OUT=HB3_${TAG}_nbx${L}; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l); echo "-- $OUT: $n/2056"
    [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
  done
done
echo HB3NBX_PACK_DONE $(date +%H:%M)
