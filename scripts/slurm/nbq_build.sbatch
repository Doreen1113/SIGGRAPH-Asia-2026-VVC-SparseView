#!/bin/bash
#SBATCH --job-name=nbq_build
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/nbq_build_%j.log
# Pre-build test-scene candidates for the person-source question so the 23:17 window can use whichever val picks.
# All use the new projected background NB_pAVt15.
#   NBQ_nb : person source = the diverse mix itself (NB_TTA over NB_ens)
#   NBQ_hb : person source = half S6E (S5-dominated) + half NB (diverse)
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY $W/tta_average.py $V/submissions/HB_ens/renders $V/submissions/S6E_ens/renders $V/submissions/NB_ens/renders --procs 90
$PY $W/tta_average.py $V/submissions/HB_TTA/renders $V/submissions/S6E_TTA/renders $V/submissions/NB_TTA/renders --procs 90
echo "HB_ens $(find $V/submissions/HB_ens -name '*.jpg' | wc -l)  HB_TTA $(find $V/submissions/HB_TTA -name '*.jpg' | wc -l)"
i=0
for spec in "nb NB_ens NB_TTA 0.85" "hb HB_ens HB_TTA 0.85" "hb HB_ens HB_TTA 0.70"; do
  set -- $spec; TAG=$(printf "p%03d" $(echo "$4*100/1" | bc)); OUT=NBQ_$1_$TAG
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/$2/renders $V/submissions/$3/renders $4 \
      $V/submissions/NB_pAVt15/renders 1.0 $V/submissions/$OUT/renders --cases $c > $V/jobs/nbqcomp_${OUT}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for OUT in NBQ_nb_p085 NBQ_hb_p085 NBQ_hb_p070; do
  n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l); echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo NBQ_BUILD_DONE $(date +%H:%M)
