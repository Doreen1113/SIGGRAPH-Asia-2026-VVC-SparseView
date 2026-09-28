#!/bin/bash
#SBATCH --job-name=val_dec
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_dec_%j.log
# The submitted SIX_p070 landed at FULL-SSIM margin +0.0004 and FULL-LPIPS short by 0.0012 - both ends of the
# same lambda knob, so tuning it cannot win both. Build the exact val analogue of that package, then the same
# recipe on the DFX base at several lambdas, and compare like for like.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# A) val analogue of the SUBMITTED package: peer base, Difix-TTA person at 0.70, projection lam15 background
i=0
for PA in 0.60 0.70 0.80; do
  TAG=$(printf "sub%03d" $(echo "$PA*100/1" | bc))
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $D repo199tta $PA pAVt_l15 1.0 $TAG > /dev/null 2>&1 &
  i=$((i+1))
done
wait
echo "=== A) 已送出版本的 val 對照（peer 基底, lam15）==="
$PY $W/metrics_both.py $D sub060 sub070 sub080 2>&1 | grep -vE "Warning|warn|load_state"
# B) same recipe on the DFX base at lam 10/15/20
S=$D/prod_dfx
for L in 10 15; do
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $S repo199tta pAVt_l$L --nets alex:1,vgg:0.5 --lam $L --steps 200 --batch 2 --shard $i/8 > /dev/null 2>&1 & done; wait
done
i=0
for L in 10 15 20; do
  for PA in 0.60 0.70 0.80; do
    TAG=$(printf "d%02d_%03d" $L $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S repo199tta $PA pAVt_l$L 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
  done
done
wait
echo "=== B) DFX 基底、同配方、三種 lambda ==="
$PY $W/metrics_both.py $S d10_060 d10_070 d10_080 d15_060 d15_070 d15_080 d20_060 d20_070 d20_080 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_DECISIVE_DONE
