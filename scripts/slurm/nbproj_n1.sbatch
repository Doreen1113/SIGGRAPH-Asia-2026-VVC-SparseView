#!/bin/bash
#SBATCH --job-name=nbproj1
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/nbproj_n1_%j.log
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q NB_TTA_DONE $V/jobs/nb_tta_comp_*.log 2>/dev/null; do sleep 60; done
for g in 0 1 2 3 4 5 6 7; do
  S=$(( 1*8+g ))
  CUDA_VISIBLE_DEVICES=$g $PY $W/proj_multi.py tree $V/submissions/NB_ens/renders $V/submissions/NB_TTA/renders     $V/submissions/NB_pAVt15/renders --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $S/32 > $V/jobs/nbproj_$S.log 2>&1 &
done; wait
echo "node 1: NB_pAVt15 $(find $V/submissions/NB_pAVt15 -name '*.jpg' | wc -l)/2056"; echo NBPROJ_N1_DONE $(date +%H:%M)
