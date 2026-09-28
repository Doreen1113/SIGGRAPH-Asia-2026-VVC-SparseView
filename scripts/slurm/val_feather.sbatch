#!/bin/bash
#SBATCH --job-name=val_feather
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_feather_%j.log
# Never-swept micro-lever: the soft-mask blur kernel width in composite2.py (hardcoded 25px until now).
# Self-contained restage (old dil_s5 dir was lost in the quota cleanup): person = S5 Difix TTA (repo199tta),
# background = bg_p4x2d2s (best previously-validated background proxy still on disk), alpha 0.85.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0; PY=/home/doreen071/.conda/envs/vvc/bin/python
S=$D/ps_feather; mkdir -p $S; cp $D/manifest.json $S/
for f in $D/*_gt.png $D/*_ref.png $D/*_repo199tta.png $D/*_bg_p4x2d2s.png; do
  b=$(basename $f); t=${b%_*}
  case "$b" in *_repo199tta.png) ln -sf $f $S/${t}_tta.png;; *_bg_p4x2d2s.png) ln -sf $f $S/${t}_bg.png;; *) ln -sf $f $S/;; esac
done
for f in $H/m_S5_FGW40_L60K/*_render.png; do t=$(basename $f _render.png); ln -sf $f $S/${t}_render.png; done
echo "staged: gt=$(ls $S/*_gt.png|wc -l) tta=$(ls $S/*_tta.png|wc -l) bg=$(ls $S/*_bg.png|wc -l) render=$(ls $S/*_render.png|wc -l)"
i=0
for FW in 9 15 25 35 45 61; do
  TAG=f$(printf "%02d" $FW)
  CUDA_VISIBLE_DEVICES=$((i%4)) $PY $W/composite2.py dump $S tta 0.85 bg 1.0 $TAG --feather $FW > $V/jobs/valfeather_$TAG.log 2>&1 &
  i=$((i+1)); [ $((i%4)) -eq 0 ] && wait
done
wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(f25 = 現行寬度,對照組)"
$PY $W/metrics_both.py $S f09 f15 f25 f35 f45 f61 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_FEATHER_DONE $(date +%H:%M)
