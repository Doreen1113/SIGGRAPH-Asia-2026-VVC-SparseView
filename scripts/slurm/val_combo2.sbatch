#!/bin/bash
#SBATCH --job-name=val_combo2
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_combo2_%j.log
# Joint check: HB3 person source (half S5 + diverse-with-S3) over the cross-target new background.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
S=$D/ps_bg_p4d2s3_h; P=$D/ps_bg_p4x2d2s
g=0
for v in xl10 xl15; do
  for f in $P/*_$v.png; do t=$(basename $f _$v.png); ln -sf $f $S/${t}_bg_$v.png; done
  for PA in 0.70 0.85; do
    TAG=$(printf "p%03d" $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$g $PY $W/composite2.py dump $S tta $PA bg_$v 1.0 j_${TAG}_$v > /dev/null 2>&1 &
    g=$((g+1))
  done
done
wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(對照 d00 現行: 26.478/0.9274/0.1478/0.3085    24.648/0.8675/0.1934/0.3246)"
$PY $W/metrics_both.py $S j_p070_xl10 j_p085_xl10 j_p070_xl15 j_p085_xl15 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_COMBO2_DONE $(date +%H:%M)
