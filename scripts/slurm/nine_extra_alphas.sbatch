#!/bin/bash
#SBATCH --job-name=nine_alphas
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/nine_alphas_%j.log
# Extra person-alpha candidates on the 9-member base. FG-SSIM and FG-LPIPS pull against each other via person
# alpha; the 9-member ensemble's ~+0.004 FG-SSIM headroom may let alpha go higher while still clearing shengqi.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
SRC=$V/submissions/NINE_ens/renders
echo "== waiting for projection tree to finish, $(date -u +%H:%M) =="
until [ "$(find $V/submissions/NINE_pAVt15 -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ]; do sleep 120; done
echo "== compositing extra alphas, $(date -u +%H:%M) =="
i=0
for PA in 0.70 0.80; do
  OUT=$V/submissions/NINE_p0${PA:2:2}_bpAVt15/renders
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $SRC $V/submissions/NINE_TTA/renders $PA \
      $V/submissions/NINE_pAVt15/renders 1.0 $OUT --cases $c > $V/jobs/nine_a${PA}_$c.log 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for PA in 70 80; do
  OUT=NINE_p0${PA}_bpAVt15
  echo "-- $OUT: $(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)/2056"
  $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders
done
echo NINE_ALPHAS_DONE $(date -u +%H:%M)
