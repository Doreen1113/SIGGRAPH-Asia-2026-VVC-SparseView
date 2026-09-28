#!/bin/bash
#SBATCH --job-name=render_dfx2
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/render_dfx2_%j.log
# Render the round-2 distilled models on the test scenes so they are ready the moment val confirms the gain.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
V=/work/doreen071/vvc; W=$V/work; i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  OUT=$V/submissions/M_DFX2_s0/renders/$c; NEED=$(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' | wc -l)
  [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ] && { echo "complete $c"; continue; }
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$((i%8)) python $W/render_hidden.py $V/runs/siga_$c/run_DFX2_s0 $V/data/$c $OUT \
      --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_DFX2_s0_$c.log 2>&1 ) &
  i=$((i+1))
done
wait
echo "M_DFX2_s0: $(find $V/submissions/M_DFX2_s0 -name '*.jpg' | wc -l)/2056"
echo RENDER_DFX2_DONE $(date +%H:%M)
