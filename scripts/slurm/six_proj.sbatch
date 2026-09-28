#!/bin/bash
#SBATCH --job-name=six_proj
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/six_proj_%x_%j.log
# The 9-member base measured WORSE than the 6-member on the real test set (all six metrics), so apply the
# dual-backbone projection to the ORIGINAL 6-member base instead. Usage: --export=ALL,BASE=0..3 (32-way split)
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in 0 1 2 3 4 5 6 7; do
  S=$((BASE*8+i))
  CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py tree $V/submissions/SIX_ens/renders $V/submissions/SIX_TTA/renders \
    $V/submissions/SIX_pAVt15/renders --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $S/32 \
    > $V/jobs/sixproj_$S.log 2>&1 &
done
wait
echo "SIX_PROJ_${BASE}_DONE $(date -u +%H:%M) total=$(find $V/submissions/SIX_pAVt15 -name '*.jpg' | wc -l)/2056"
