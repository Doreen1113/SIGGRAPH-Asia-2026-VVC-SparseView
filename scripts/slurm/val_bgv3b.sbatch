#!/bin/bash
#SBATCH --job-name=val_bgv3b
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_bgv3b_%j.log
# v3 mixes again, but with a SHARP Difix target: D = S5 Difix TTA (rebuildable on test in ~1.2 h) and D = peer TTA (ceiling, not deployable).
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; C=$D/ps_cur; P=$D/ps_bgv3; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in $C/*_tta.png; do t=$(basename $f _tta.png); ln -sf $f $P/${t}_s5tta.png; done
for f in $D/*_repo199tta.png; do t=$(basename $f _repo199tta.png); ln -sf $f $P/${t}_ptta.png; done
for spec in "ownT s5tta 15 ts15" "own6nn s5tta 15 ns15" "ownT ptta 15 tp15" "ownT s5tta 10 ts10" "own6 s5tta 15 6s15" "ownT s5tta 20 ts20"; do
  set -- $spec; mix=$1; tgt=$2; lam=$3; out=$4
  for f in $P/*_$mix.png; do t=$(basename $f _$mix.png); ln -sf $f $P/${t}_render.png; done
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $P $tgt x_$out --nets alex:1,vgg:0.5 --lam $lam --steps 200 --batch 2 --shard $i/8 > $V/jobs/bgv3b_${out}_$i.log 2>&1 & done; wait
  echo "  $out ($mix -> $tgt lam$lam) $(ls $P/*_x_$out.png | wc -l)/24 $(date +%H:%M)"
done
g=0
for out in ts15 ns15 tp15 ts10 6s15 ts20; do
  for f in $P/*_x_$out.png; do t=$(basename $f _x_$out.png); ln -sf $f $C/${t}_bg_x_$out.png; done
  CUDA_VISIBLE_DEVICES=$((g%8)) $PY $W/composite2.py dump $C tta 0.85 bg_x_$out 1.0 c_x_$out > $V/jobs/bgv3b_comp_$out.log 2>&1 &
  g=$((g+1))
done; wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(c_xl15 現行;ts15 = ownT→S5 TTA λ15(可部署);ns15 = own6nn→S5 TTA;tp15 = ownT→peer TTA(上限,不可部署);ts10/ts20 = λ;6s15 = own6→S5 TTA)"
$PY $W/metrics_both.py $C c_xl15 c_x_ts15 c_x_ns15 c_x_6s15 c_x_ts10 c_x_ts20 c_x_tp15 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_BGV3B_DONE $(date +%H:%M)
