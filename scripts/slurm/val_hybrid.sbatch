#!/bin/bash
#SBATCH --job-name=val_hybrid
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_hybrid_%j.log
# Hybrid: background from the PEER base projection (best FULL-LPIPS), person from the DFX base + FGW40_L60K
# (best FG). The DFX base helps FG and PSNR but costs FULL-LPIPS, because its renders are already Difix-sharp
# and leave the projection less room; taking each side from where it is strongest should dominate both.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; P=$D/pp_ps_f40l
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
H=$D/hybrid; mkdir -p $H; cp $P/manifest.json $H/
for f in $D/*_gt.png $D/*_ref.png; do ln -sf $f $H/; done
# person source (raw, for the (1-pa) term) and its Difix-TTA
for f in $P/*_render.png; do ln -sf $f $H/; done
for f in $P/*_repo199tta.png; do ln -sf $f $H/; done
# background = the PEER base projection at lambda 20
for f in $D/*_pAVt_l20.png; do ln -sf $f $H/; done
i=0
for PA in 0.40 0.50 0.65; do
  TAG=$(printf "h%03d" $(echo "$PA*100/1" | bc))
  CUDA_VISIBLE_DEVICES=$((i%4)) $PY $W/composite2.py dump $H repo199tta $PA pAVt_l20 1.0 $TAG > /dev/null 2>&1 &
  i=$((i+1))
done
wait
echo "=== HYBRID: person = DFXbase+FGW40_L60K (Difix-TTA), background = PEER-base projection lam20 ==="
$PY $W/metrics_both.py $H h040 h050 h065 2>&1 | grep -vE "Warning|warn|load_state"
echo "--- 對照 ---"
echo "peer-base 全套 (q_f8l2 h050): 26.438/0.9275/0.1500 | 24.460/0.8688/0.2137"
echo "DFX-base 全套 (ps_f40l q050): 26.996/0.9300/0.1563 | 24.700/0.8719/0.2080"
echo VAL_HYBRID_DONE
