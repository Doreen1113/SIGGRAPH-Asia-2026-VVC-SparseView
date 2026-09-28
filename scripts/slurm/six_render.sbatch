#!/bin/bash
#SBATCH --job-name=six_render
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/six_render_%j.log
# Rebuild SIX_ens (equal-weight 6-member peer ensemble) on all 5 test scenes now that the peer sent back the
# checkpoints (FULLRES_s0, P8_s0, P9_s0, J_s0, GATE_s0, G4M_s0).
V=/work/doreen071/vvc; W=$V/work
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  R=$V/runs/siga_$c; OUT=$V/submissions/SIX_ens/renders/$c
  [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -gt 0 ] && { echo "skip $c"; continue; }
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$i python $W/render_hidden.py $R/run_FULLRES_s0 $V/data/$c $OUT \
      --cull_near_frac 1.1 --cull_min_views 4 --extra_runs $R/run_P8_s0,$R/run_P9_s0,$R/run_J_s0,$R/run_GATE_s0,$R/run_G4M_s0 \
      > $V/jobs/six_render_$c.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1))
done
wait
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  echo "$c: $(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' 2>/dev/null | wc -l)"
done
echo SIX_RENDER_DONE $(date +%H:%M)
