#!/bin/bash
#SBATCH --job-name=val_bgv3c
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_bgv3c_%j.log
# Last peer-containing targets that still exist on test: SIX_pAVt15 / SIX_pAVt20 (val: pAVt_l15 / pAVt_l20).
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; C=$D/ps_cur; P=$D/ps_bgv3; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in $D/*_pAVt_l15.png; do t=$(basename $f _pAVt_l15.png); ln -sf $f $P/${t}_pl15.png; done
for f in $D/*_pAVt_l20.png; do t=$(basename $f _pAVt_l20.png); ln -sf $f $P/${t}_pl20.png; done
for spec in "ownT pl15 15 tq15" "ownT pl20 15 tr15" "ownT pl15 8 tq08" "own6nn pl15 15 nq15"; do
  set -- $spec; mix=$1; tgt=$2; lam=$3; out=$4
  for f in $P/*_$mix.png; do t=$(basename $f _$mix.png); ln -sf $f $P/${t}_render.png; done
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $P $tgt x_$out --nets alex:1,vgg:0.5 --lam $lam --steps 200 --batch 2 --shard $i/8 > $V/jobs/bgv3c_${out}_$i.log 2>&1 & done; wait
  echo "  $out ($mix -> $tgt lam$lam) $(ls $P/*_x_$out.png | wc -l)/24 $(date +%H:%M)"
done
g=0
for out in tq15 tr15 tq08 nq15; do
  for f in $P/*_x_$out.png; do t=$(basename $f _x_$out.png); ln -sf $f $C/${t}_bg_x_$out.png; done
  CUDA_VISIBLE_DEVICES=$((g%8)) $PY $W/composite2.py dump $C tta 0.85 bg_x_$out 1.0 c_x_$out > $V/jobs/bgv3c_comp_$out.log 2>&1 &
  g=$((g+1))
done; wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(c_xl15 現行;tq15 = ownT→SIX_pAVt15 λ15;tr15 = →SIX_pAVt20;tq08 = λ8;nq15 = own6nn→SIX_pAVt15;tp15 = peer TTA 上限)"
$PY $W/metrics_both.py $C c_xl15 c_x_tq15 c_x_tr15 c_x_tq08 c_x_nq15 c_x_tp15 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_BGV3C_DONE $(date +%H:%M)
