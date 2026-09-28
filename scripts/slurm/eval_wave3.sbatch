#!/bin/bash
#SBATCH --job-name=eval_wave3
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:6
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/eval_wave3_%j.log
# Evaluate the wave3 training-side experiments on val 001_1: does FG loss weight / 8M capacity / 60k steps
# actually move the metrics we are short on (FG-PSNR/SSIM/LPIPS)? Each member is rendered SOLO so the effect
# is not diluted by ensembling.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10
V=/work/doreen071/vvc; S=001_1_seq0; R=$V/runs/siga_$S
MEMBERS="FGW05_s0 FGW10_s0 FGW20_s0 G8M_FR_s0 LONG60K_s0 FGW10_G8M"
echo "== waiting for wave3 training, $(date -u +%H:%M) =="
until (for M in $MEMBERS; do [ -f $R/run_$M/gaussians.pt ] || exit 1; done); do sleep 120; done
echo "== rendering each solo, $(date -u +%H:%M) =="
i=0
for M in $MEMBERS; do
  D=$V/fixtest_h200/$S/m_$M
  if [ ! -f $D/manifest.json ]; then
    ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
      TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$((i%6)) python $V/work/dump_val_renders_h200.py $S $R/run_$M "" $D --every 200 > $V/jobs/evalw3_$M.log 2>&1
      rm -rf $(dirname $TD) ) &
    i=$((i+1))
  fi
done
wait
echo "== solo metrics (baseline = the plain full-res member FULLRES_s0) =="
PY=/home/doreen071/.conda/envs/vvc/bin/python
[ -f $V/fixtest_h200/$S/m_FULLRES_s0/manifest.json ] || python $V/work/dump_val_renders_h200.py $S $R/run_FULLRES_s0 "" $V/fixtest_h200/$S/m_FULLRES_s0 --every 200 > /dev/null 2>&1
for M in FULLRES_s0 $MEMBERS; do
  D=$V/fixtest_h200/$S/m_$M
  [ -f $D/manifest.json ] || continue
  printf "%-12s " $M
  $PY $V/work/metrics_both.py $D render 2>&1 | grep -vE "Warning|warn|load_state|variant" | tail -1
done
echo EVAL_WAVE3_DONE $(date -u +%H:%M)
