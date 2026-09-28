#!/bin/bash
#SBATCH --job-name=nine_proj
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/nine_proj_%j.log
# Dual-backbone projection over the 9-member ensemble. batch=2 (batch=4 OOMs: alex+vgg at 4K needs ~140GB).
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SHARD=${SHARD:-8}
for i in $(seq 0 $((SHARD-1))); do
  CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py tree $V/submissions/NINE_ens/renders $V/submissions/NINE_TTA/renders \
    $V/submissions/NINE_pAVt15/renders --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $i/$SHARD \
    > $V/jobs/nineproj_$i.log 2>&1 &
done
wait
echo "projection: $(find $V/submissions/NINE_pAVt15 -name '*.jpg' | wc -l)/2056"
echo NINE_PROJ_DONE $(date -u +%H:%M)
