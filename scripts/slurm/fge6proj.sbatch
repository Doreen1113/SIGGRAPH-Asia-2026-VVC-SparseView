#!/bin/bash
#SBATCH --job-name=fge6proj
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fge6proj_%j.log
# The dual-backbone perceptual projection has only ever been applied to the BACKGROUND. Board analysis says the
# one thing standing between us and 1st place is FG-LPIPS (need -0.006 to pass yunqigao's 0.23799), and the
# projection is exactly the tool that buys LPIPS. So: project the FG-ensemble person source too.
# 32-way split over 4 nodes (8 GPUs each) because the projection is the slowest step we have.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
srun --ntasks=4 --ntasks-per-node=1 bash -c '
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=$SLURM_PROCID
for g in 0 1 2 3 4 5 6 7; do
  S=$((N*8+g))
  CUDA_VISIBLE_DEVICES=$g $PY $W/proj_multi.py tree $V/submissions/FGE6_ens/renders $V/submissions/FGE6_TTA/renders \
    $V/submissions/FGE6_pAVt15/renders --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $S/32 \
    > $V/jobs/fge6proj_$S.log 2>&1 &
done
wait
echo "node $N done $(date +%H:%M)"
'
echo "FGE6_pAVt15: $(find $V/submissions/FGE6_pAVt15 -name '*.jpg' | wc -l)/2056"
echo FGE6PROJ_DONE $(date +%H:%M)
