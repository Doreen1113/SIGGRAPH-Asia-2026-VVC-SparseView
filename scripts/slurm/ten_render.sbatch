#!/bin/bash
#SBATCH --job-name=ten_render
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:5
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/ten_render_%j.log
# Render FGW20_s0 on the 5 test scenes, then build the 10-member ensemble with FGW20 at weight 3
# (val: weight 3 beats weight 1 on FG-LPIPS by 0.007 at equal-or-better FG-SSIM -> predicted rank 1.833 vs 2.000).
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10
V=/work/doreen071/vvc
echo "== waiting for FGW20_s0 on all 5 test scenes, $(date -u +%H:%M) =="
until (for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
         [ -f $V/runs/siga_$c/run_FGW20_s0/gaussians.pt ] || exit 1; done); do sleep 60; done
echo "== rendering FGW20, $(date -u +%H:%M) =="
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  OUT=$V/submissions/M_FGW20_s0/renders/$c
  NEED=$(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' | wc -l)
  [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ] && { echo "complete $c"; continue; }
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$i python $V/work/render_hidden.py $V/runs/siga_$c/run_FGW20_s0 $V/data/$c $OUT \
      --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_FGW20_$c.log 2>&1
    rm -rf $(dirname $TD) ) &
  i=$((i+1))
done
wait
echo "FGW20 renders: $(find $V/submissions/M_FGW20_s0 -name '*.jpg' | wc -l)/2056"
echo "== 10-member weighted ensemble (FGW20 at weight 3), $(date -u +%H:%M) =="
/home/doreen071/.conda/envs/vvc/bin/python $V/work/weighted_ensemble.py $V/submissions/TEN_ens/renders \
  $V/submissions/SIX_ens/renders:6 $V/submissions/M_G3M_FR_s0/renders:1 \
  $V/submissions/M_GATE_FR_s0/renders:1 $V/submissions/M_FULLRES_s1/renders:1 \
  $V/submissions/M_FGW20_s0/renders:3 --procs 60
echo TEN_RENDER_DONE $(date -u +%H:%M)
