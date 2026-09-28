#!/bin/bash
#SBATCH --job-name=lam_margin
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/lam_margin_%j.log
# Our two SSIM margins are razor thin while both LPIPS margins are wide. The projection's lambda trades exactly
# that way (higher lambda = closer to the raw render = more SSIM). Sweep it to maximise the WORST margin.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
i=0
for L in 10 20 25 30; do
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/proj_multi.py dump $D repo199tta pAVt_l$L --nets alex:1,vgg:0.5 --lam $L --steps 200 --batch 2 > $V/jobs/lam_$L.log 2>&1 &
  i=$((i+1))
done
wait
# person source stays the best mix (FGW80x2 + LONG60Kx2); only the background projection lambda changes
for L in 10 15 20 25 30; do for f in $D/*_pAVt_l$L.png; do ln -sf $f $D/hunt_q_f8l2/; done; done
i=0
for L in 10 15 20 25 30; do
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $D/hunt_q_f8l2 repo199tta 0.50 pAVt_l$L 1.0 lam$L > /dev/null 2>&1 &
  i=$((i+1))
done
wait
echo "=== background projection lambda sweep (person = FGW80x2+LONG60Kx2 @ alpha .50) ==="
$PY $W/metrics_both.py $D/hunt_q_f8l2 lam10 lam15 lam20 lam25 lam30 2>&1 | grep -vE "Warning|warn|load_state"
echo LAM_MARGIN_DONE
