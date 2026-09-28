#!/bin/bash
#SBATCH --job-name=nb_build
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/nb_build_%j.log
# val_bgmix: background = peer4 + DFX_s0 x2 + DFXFG x2 + S5 x2 beats the peer-only background on all six metrics
# (FULL-PSNR +0.80 dB). Build the same 4:2:2:2 mix on the test scenes. SIX_ens is a 6-member mean, so it enters
# with weight 4 as one root; S5 enters as the mean of its 6 seeds.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
i=0
for M in DFXFG_L60K DFXFG_L60K_s1; do
 for c in $CASES; do
  OUT=$V/submissions/M_$M/renders/$c; NEED=$(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' | wc -l)
  [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ] && continue
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$((i%8)) python $W/render_hidden.py $V/runs/siga_$c/run_$M $V/data/$c $OUT \
      --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_${M}_$c.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
 done
done
wait
for M in DFXFG_L60K DFXFG_L60K_s1; do echo "$M: $(find $V/submissions/M_$M -name '*.jpg' | wc -l)/2056"; done
$PY $W/tta_average.py $V/submissions/S5P_ens/renders \
  $V/submissions/M_S5_FGW40_L60K/renders $V/submissions/M_S5_FGW40_L60K_s1/renders $V/submissions/M_S5_FGW40_L60K_s2/renders \
  $V/submissions/M_S5_FGW40_L60K_s3/renders $V/submissions/M_S5_FGW40_L60K_s4/renders $V/submissions/M_S5_FGW40_L60K_s5/renders --procs 90
$PY $W/weighted_ensemble.py $V/submissions/NB_ens/renders $V/submissions/SIX_ens/renders:4 $V/submissions/M_DFX_s0/renders:2 \
  $V/submissions/M_DFXFG_L60K/renders:1 $V/submissions/M_DFXFG_L60K_s1/renders:1 $V/submissions/S5P_ens/renders:2 --procs 90
echo "NB_ens: $(find $V/submissions/NB_ens -name '*.jpg' | wc -l)/2056"
echo NB_ENS_DONE $(date +%H:%M)
