#!/bin/bash
#SBATCH --job-name=eval_members
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:3
#SBATCH --time=03:55:00
#SBATCH --output=/work/doreen071/vvc/jobs/eval_members_%x_%j.log
# Generic: wait for the listed val members to finish training, render each solo, print metrics.
# Usage: sbatch --job-name=<tag> --export=ALL,MEMBERS="A B C" eval_members.sbatch
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10
V=/work/doreen071/vvc; S=001_1_seq0; R=$V/runs/siga_$S; PY=/home/doreen071/.conda/envs/vvc/bin/python
until (for M in $MEMBERS; do [ -f $R/run_$M/gaussians.pt ] || exit 1; done); do sleep 120; done
i=0
for M in $MEMBERS; do
  D=$V/fixtest_h200/$S/m_$M; [ -f $D/manifest.json ] && continue
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$((i%3)) python $V/work/dump_val_renders_h200.py $S $R/run_$M "" $D --every 200 > $V/jobs/evalm_$M.log 2>&1; rm -rf $(dirname $TD) ) &
  i=$((i+1)); [ $((i%3)) -eq 0 ] && wait
done
wait
echo "=== solo metrics (baseline FULLRES_s0 24.282/0.9135/0.2065 | 22.402/0.8274/0.2915 ; FGW20 25.205/0.9175/0.2020 | 23.265/0.8412/0.2806) ==="
for M in $MEMBERS; do printf "%-12s " $M; $PY $V/work/metrics_both.py $V/fixtest_h200/$S/m_$M render 2>&1 | grep -vE "Warning|warn|load_state|variant" | tail -1; done
echo EVAL_MEMBERS_DONE
