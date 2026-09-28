#!/bin/bash
#SBATCH --job-name=nbx15_0
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/nbx15_n0_%j.log
# Cross-target projection: structure from the diverse-mix raw (NB_ens), perceptual target = the PEER Difix TTA
# (SIX_TTA). val: lam15 27.052/0.9291/0.1468 | 24.885/0.8690/0.1892, all six better than the current background.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in 0 1 2 3 4 5 6 7; do
  S=$(( 0*8+g ))
  CUDA_VISIBLE_DEVICES=$g $PY $W/proj_multi.py tree $V/submissions/NB_ens/renders $V/submissions/SIX_TTA/renders     $V/submissions/NBX15_pAV/renders --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $S/32 > $V/jobs/nbx15_$S.log 2>&1 &
done; wait
echo "node 0: NBX15_pAV $(find $V/submissions/NBX15_pAV -name '*.jpg' | wc -l)/2056"; echo NBX15_N0_DONE $(date +%H:%M)
