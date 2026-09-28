#!/bin/bash
#SBATCH --job-name=fge6proj2
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fge6proj_n2_%j.log
# One independent job per node: the earlier srun-based 4-task step was killed the moment its first task finished
# (SLURM WaitTime), taking the other three nodes with it. proj_multi skips images that already exist, so this
# resumes the 1657/2056 that survived.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in 0 1 2 3 4 5 6 7; do
  S=$(( 2*8+g ))
  CUDA_VISIBLE_DEVICES=$g $PY $W/proj_multi.py tree $V/submissions/FGE6_ens/renders $V/submissions/FGE6_TTA/renders     $V/submissions/FGE6_pAVt15/renders --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $S/32     >> $V/jobs/fge6proj_$S.log 2>&1 &
done
wait
echo "node 2: FGE6_pAVt15 now $(find $V/submissions/FGE6_pAVt15 -name '*.jpg' | wc -l)/2056"
echo FGE6PROJ_N2_DONE $(date +%H:%M)
