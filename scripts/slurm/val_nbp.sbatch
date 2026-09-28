#!/bin/bash
#SBATCH --job-name=val_nbp
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_nbp_%j.log
# Answer "does NBP improve all six?" on val before the 23:17 window: same recipe as the test package
# (person = S5 TTA alpha 0.85 over S5 raw; background = projection of the diverse mix), versus the current
# recipe (same person, background = projection of the peer ensemble).
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$D/ps_bg_p4x2d2s; S=$D/dil_s5
until [ "$(ls $P/*_tta.png 2>/dev/null | wc -l)" -eq 24 ]; do sleep 30; done
echo "== project the new background (lam15, alex:1 vgg:0.5, 200 steps), $(date +%H:%M) =="
CUDA_VISIBLE_DEVICES=0 $PY $W/proj_multi.py dump $P tta nbproj --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 > $V/jobs/valnbp_proj.log 2>&1
n=0; for f in $P/*_nbproj.png; do t=$(basename $f _nbproj.png); ln -sf $f $S/${t}_nbg.png; n=$((n+1)); done; echo "projected new bg: $n"
CUDA_VISIBLE_DEVICES=1 $PY $W/composite2.py dump $S tta 0.85 nbg 1.0 nbp > $V/jobs/valnbp_comp.log 2>&1
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(d00 = 現行配方: S5人物 + peer投影背景;  nbp = NBP配方: S5人物 + 新背景投影)"
$PY $W/metrics_both.py $S d00 nbp 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_NBP_DONE $(date +%H:%M)
