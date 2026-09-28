#!/bin/bash
#SBATCH --job-name=val_proj
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_proj_%j.log
PY=/home/doreen071/.conda/envs/vvc/bin/python
W=/work/doreen071/vvc/work; F=/work/doreen071/vvc/fixtest
export OMP_NUM_THREADS=6
run() { CUDA_VISIBLE_DEVICES=$1 $PY $W/proj_multi.py dump $F/$2 $3 $4 --nets "$5" --lam $6 --steps 200 --batch 2 > /work/doreen071/vvc/jobs/valproj_$2_$4.log 2>&1 & }
run 0 001_1 repo199    pAV_l10  alex:1,vgg:0.5 10
run 1 001_1 repo199    pAV_l15  alex:1,vgg:0.5 15
run 2 001_1 repo199tta pAVt_l10 alex:1,vgg:0.5 10
run 3 001_1 repo199tta pAVt_l15 alex:1,vgg:0.5 15
run 4 012_0 repo199    pAV_l10  alex:1,vgg:0.5 10
run 5 012_0 repo199    pAV_l15  alex:1,vgg:0.5 15
run 6 012_0 repo199tta pAVt_l10 alex:1,vgg:0.5 10
run 7 012_0 repo199tta pAVt_l15 alex:1,vgg:0.5 15
wait
for S in 001_1 012_0; do
  CUDA_VISIBLE_DEVICES=0 $PY $W/composite2.py dump $F/$S repo199tta 0.50 pAVt_l10 1.0 pbt_p050_bpAVt10
  CUDA_VISIBLE_DEVICES=1 $PY $W/composite2.py dump $F/$S repo199tta 0.50 pAVt_l15 1.0 pbt_p050_bpAVt15
  CUDA_VISIBLE_DEVICES=2 $PY $W/composite2.py dump $F/$S repo199tta 0.55 pAVt_l10 1.0 pbt_p055_bpAVt10
  CUDA_VISIBLE_DEVICES=0 $PY $W/metrics_both.py $F/$S render repo199ttaa045 pbt_p050_b045 pb_p060_b055 proj_l10 pAV_l10 pAV_l15 pAVt_l10 pAVt_l15 pbt_p050_bpAVt10 pbt_p050_bpAVt15 pbt_p055_bpAVt10 2>&1 | grep -vE "Warning|warn|load_state" > /work/doreen071/vvc/jobs/metricsAV_$S.log
done
echo VAL_PROJ_DONE
