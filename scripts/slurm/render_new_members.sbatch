#!/bin/bash
#SBATCH --job-name=render_new
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/render_new_%j.log
# Render each NEW H200-trained member separately on all 5 test scenes (culled, same protocol as SIX_ens),
# so they can be weight-averaged with the existing 6-member SIX_ens render tree.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10
V=/work/doreen071/vvc
MEMBERS="G3M_FR_s0 GATE_FR_s0 FULLRES_s1 G4M_FR_s0"
echo "waiting for all members to finish training..."
until (for C in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
         for M in $MEMBERS; do [ -f $V/runs/siga_$C/run_$M/gaussians.pt ] || exit 1; done
       done); do sleep 60; done
echo "all members present, rendering $(date -u +%H:%M)"
i=0
for M in $MEMBERS; do
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    OUT=$V/submissions/M_$M/renders/$c
    NEED=$(find $V/submissions/SIX_ens/renders/$c -name '*.jpg' | wc -l)
    if [ "$(find $OUT -name '*.jpg' 2>/dev/null | wc -l)" -eq "$NEED" ]; then echo "complete $M/$c"; continue; fi
    ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD
      cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
      TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$((i%8)) python $V/work/render_hidden.py $V/runs/siga_$c/run_$M $V/data/$c $OUT \
        --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_${M}_$c.log 2>&1
      rm -rf $(dirname $TD) ) &
    i=$((i+1))
    [ $((i%8)) -eq 0 ] && wait
  done
done
wait
for M in $MEMBERS; do echo "$M: $(find $V/submissions/M_$M -name '*.jpg' 2>/dev/null | wc -l) / 2056"; done
echo RENDER_NEW_DONE $(date -u +%H:%M)
