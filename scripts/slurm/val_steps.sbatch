#!/bin/bash
#SBATCH --job-name=val_steps
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_steps_%j.log
# Projection optimisation steps have never been tuned (always 200). More steps converge closer to the LPIPS
# target at the SAME lambda, so this should lower LPIPS without spending SSIM - which is exactly the shape of
# improvement we need (FULL-LPIPS short by 0.0012, FULL-SSIM margin only 0.0004).
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for N in 400 800; do
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $D repo199tta pAVs${N} --nets alex:1,vgg:0.5 --lam 15 --steps $N --batch 2 --shard $i/8 > /dev/null 2>&1 &
  done
  wait; echo "steps=$N done $(date -u +%H:%M)"
done
i=0
for N in 400 800; do
  for PA in 0.70 0.80; do
    TAG=$(printf "s%s_%03d" $N $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $D repo199tta $PA pAVs${N} 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1))
  done
done
wait
echo "=== 投影步數掃描（lam15 固定；對照 = 已送出的 200 步版本 sub070/sub080）==="
$PY $W/metrics_both.py $D sub070 sub080 s400_070 s400_080 s800_070 s800_080 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_STEPS_DONE
