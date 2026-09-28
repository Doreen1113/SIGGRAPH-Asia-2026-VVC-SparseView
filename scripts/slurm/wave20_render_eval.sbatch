#!/bin/bash
#SBATCH --job-name=w20_render
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/w20_render_%j.log
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
i=0
for m in P15_011w50 P16_011w60 P17_011gate P18_011fr P19_011damp; do
  OUT=$S/M_$m/renders/011_0_seq0
  [ "$(cnt $OUT)" -gt 0 ] && { i=$((i+1)); continue; }
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$i python $W/render_hidden.py $V/runs/siga_011_0_seq0/run_$m $V/data/011_0_seq0 $OUT --cull_near_frac 1.1 --cull_min_views 4 > $V/jobs/renderM_${m}_011.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1))
done; wait
for m in P15_011w50 P16_011w60 P17_011gate P18_011fr P19_011damp; do echo "M_$m: $(cnt $S/M_$m/renders/011_0_seq0)/360"; done
echo W20_RENDER_DONE $(date +%H:%M)
