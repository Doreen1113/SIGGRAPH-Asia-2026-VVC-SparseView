#!/bin/bash
#SBATCH --job-name=nb_tta_comp
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/nb_tta_comp_%j.log
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until (for d in NB_difix NB_difix_s1 NB_difix_s2; do [ "$(find $V/submissions/$d -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ] || exit 1; done); do sleep 60; done
$PY $W/tta_average.py $V/submissions/NB_TTA/renders $V/submissions/NB_difix/renders $V/submissions/NB_difix_s1/renders $V/submissions/NB_difix_s2/renders --procs 90
echo "NB_TTA: $(find $V/submissions/NB_TTA -name '*.jpg' | wc -l)/2056"; echo NB_TTA_DONE $(date +%H:%M)
until [ "$(find $V/submissions/S6E_TTA -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ]; do sleep 60; done
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/S6E_ens/renders $V/submissions/S6E_TTA/renders 0.85 \
    $V/submissions/NB_TTA/renders 1.0 $V/submissions/NBT_p085/renders --cases $c > $V/jobs/nbtcomp_$c.log 2>&1 &
  i=$((i+1))
done; wait
n=$(find $V/submissions/NBT_p085 -name '*.jpg' 2>/dev/null | wc -l); echo "-- NBT_p085: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/NBT_p085.zip $V/submissions/NBT_p085/renders && chmod 444 $V/submissions/NBT_p085.zip
echo NBT_PACK_DONE $(date +%H:%M)
