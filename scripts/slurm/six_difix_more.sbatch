#!/bin/bash
#SBATCH --job-name=six_dfx_more
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/six_dfx_more_%x_%j.log
# Three more tile-grid shifts for the Difix self-ensemble. TTA is variance reduction, not an SSIM/LPIPS trade,
# so more shifts should lower LPIPS at ~unchanged SSIM - exactly the 0.0012 we are short on FULL-LPIPS.
V=/work/doreen071/vvc; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
SHIFT="$DX,$DY"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $V/work/difix_submission_h200.py $V/submissions/SIX_ens/renders \
    $V/submissions/$OUT/renders --shift $SHIFT > $V/jobs/sixmore_${OUT}_$i.log 2>&1 &
done
wait
echo "$OUT: $(find $V/submissions/$OUT -name '*.jpg' | wc -l)/2056  SIX_MORE_${OUT}_DONE $(date -u +%H:%M)"
