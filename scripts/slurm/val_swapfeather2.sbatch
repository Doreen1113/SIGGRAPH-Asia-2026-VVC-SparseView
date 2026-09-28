#!/bin/bash
#SBATCH --job-name=val_swapft2
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=00:40:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_swapfeather2_%j.log
V=/work/doreen071/vvc; W=$V/work; C=$V/fixtest/001_1/ps_cur; PY=/home/doreen071/.conda/envs/vvc/bin/python
g=0
for spec in "81 2 0.15 0.5 s2" "81 4 0.15 0.5 s4" "81 8 0.15 0.5 s8" "81 4 0.05 0.3 s4w" "81 0 0.3 0.7 cons" "61 4 0.15 0.5 s4k61" "101 4 0.15 0.5 s4k101" "81 4 0.25 0.6 s4c"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=$g $PY $W/swap_feather.py dump $C cj bg_xl15 B sf_$5 --k1 $1 --sigma $2 --t0 $3 --t1 $4 > $V/jobs/valsf2_$5.log 2>&1 &
  g=$((g+1))
done; wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(基準 cj = 現行打包;sfB81 = 上一版反推)"
$PY $W/metrics_both.py $C cj sfB81 sf_s2 sf_s4 sf_s8 sf_s4w sf_cons sf_s4c sf_s4k61 sf_s4k101 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_SWAPFT2_DONE $(date +%H:%M)
