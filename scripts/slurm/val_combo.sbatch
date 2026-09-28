#!/bin/bash
#SBATCH --job-name=val_combo
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_combo_%j.log
# Full combination on val: person source = half S5 + half diverse-with-S3 (TTA), background = projected diverse mix.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
until grep -q VAL_NBP_DONE $V/jobs/val_nbp_*.log 2>/dev/null; do sleep 30; done
S=$D/ps_bg_p4d2s3_h
for f in $D/ps_bg_p4x2d2s/*_nbproj.png; do t=$(basename $f _nbproj.png); ln -sf $f $S/${t}_nbg.png; done
CUDA_VISIBLE_DEVICES=0 $PY $W/composite2.py dump $S tta 0.70 nbg 1.0 c070 > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $W/composite2.py dump $S tta 0.85 nbg 1.0 c085 > /dev/null 2>&1 &
wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(對照 d00 現行: 26.478/0.9274/0.1478/0.3085    24.648/0.8675/0.1934/0.3246)"
$PY $W/metrics_both.py $S p070 p085 c070 c085 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_COMBO_DONE $(date +%H:%M)
