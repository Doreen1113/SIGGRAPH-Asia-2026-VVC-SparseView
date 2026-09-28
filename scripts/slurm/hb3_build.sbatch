#!/bin/bash
#SBATCH --job-name=hb3_build
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/hb3_build_%j.log
# val_personsrc: person source = half S5-family + half diverse mix INCLUDING S3 improves all six over the S5 person
# source (p070: FG-SSIM +0.0046). Build the test-scene equivalent: NB3 = SIX:4 DFX:2 DFXFG:2 S5P:2 S3:2, then
# HB3 = mean(S6E_ens, NB3_ens). Difix runs in three separate node jobs (difix_HB3_*).
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  OUT=$V/submissions/M_S3_FGW40_L60K/renders/$c; NEED=$(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' | wc -l)
  [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ] && continue
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$((i%8)) python $W/render_hidden.py $V/runs/siga_$c/run_S3_FGW40_L60K $V/data/$c $OUT \
      --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_S3_$c.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1))
done
wait
echo "M_S3_FGW40_L60K: $(find $V/submissions/M_S3_FGW40_L60K -name '*.jpg' | wc -l)/2056"
$PY $W/weighted_ensemble.py $V/submissions/NB3_ens/renders $V/submissions/SIX_ens/renders:4 $V/submissions/M_DFX_s0/renders:2 \
  $V/submissions/M_DFXFG_L60K/renders:1 $V/submissions/M_DFXFG_L60K_s1/renders:1 $V/submissions/S5P_ens/renders:2 \
  $V/submissions/M_S3_FGW40_L60K/renders:2 --procs 90
$PY $W/tta_average.py $V/submissions/HB3_ens/renders $V/submissions/S6E_ens/renders $V/submissions/NB3_ens/renders --procs 90
echo "HB3_ens: $(find $V/submissions/HB3_ens -name '*.jpg' | wc -l)/2056"; echo HB3_ENS_DONE $(date +%H:%M)
