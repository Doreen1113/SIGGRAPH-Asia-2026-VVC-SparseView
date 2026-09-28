#!/bin/bash
#SBATCH --job-name=hb3old_pack
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/hb3old_pack_%j.log
# val_personsrc: person = half S5-family + half diverse-with-S3, over the PEER projected background, improves all
# six over the S5 person source (p085 strictly; p070 ties FULL-LPIPS, FG-SSIM +0.0046). The new background is on
# hold (FULL-LPIPS regression), so composite HB3 over the proven SIX_pAVt15 background.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
i=0
for PA in 0.85 0.80 0.70; do
  TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/HB3_ens/renders $V/submissions/HB3_TTA/renders $PA \
      $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/HB3_${TAG}_six/renders --cases $c > $V/jobs/hb3oldcomp_${TAG}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for TAG in p085 p080 p070; do
  OUT=HB3_${TAG}_six; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l); echo "-- $OUT: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
done
echo HB3OLD_PACK_DONE $(date +%H:%M)
