#!/bin/bash
#SBATCH --job-name=dfxb_pw
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/dfxb_pw_%x_%j.log
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in 0 1 2 3 4 5 6 7; do
  S=$((BASE*8+i))
  CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py tree $V/submissions/DFXB_ens/renders $V/submissions/DFXB_TTA/renders \
    $V/submissions/DFXB_pAVt20/renders --nets alex:1,vgg:0.5 --lam 20 --steps 200 --batch 2 --shard $S/32 \
    > $V/jobs/dfxbpw_$S.log 2>&1 &
done
wait
echo "DFXB_PW_${BASE}_DONE $(date -u +%H:%M) total=$(find $V/submissions/DFXB_pAVt20 -name '*.jpg' | wc -l)/2056"
