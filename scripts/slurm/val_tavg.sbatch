#!/bin/bash
#SBATCH --job-name=val_tavg
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=02:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_tavg_%j.log
V=/work/doreen071/vvc; S=001_1_seq0; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
until grep -q VAL_DUMP_E10_DONE $V/jobs/val_dump_e10_*.log 2>/dev/null; do sleep 60; done
for D in s5_e10 ens5_e10; do
  DD=$V/fixtest_h200/$S/$D
  CUDA_VISIBLE_DEVICES=0 $PY $V/work/temporal_avg.py dump $DD 1 tavg1 > $V/jobs/tavg_${D}_1.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $V/work/temporal_avg.py dump $DD 2 tavg2 > $V/jobs/tavg_${D}_2.log 2>&1 &
  wait
  CUDA_VISIBLE_DEVICES=0 $PY $V/work/temporal_avg.py dump $DD 4 tavg4 > $V/jobs/tavg_${D}_4.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $V/work/temporal_avg.py dump $DD 8 tavg8 > $V/jobs/tavg_${D}_8.log 2>&1 &
  wait
  echo "=== $D: raw vs temporal avg K=1,2,4,8 (K = 相鄰提交幀數, 每幀相隔10原始幀) ==="
  echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
  $PY $V/work/metrics_both.py $DD render tavg1 tavg2 tavg4 tavg8 2>&1 | grep -vE "Warning|warn|load_state"
done
echo VAL_TAVG_DONE
