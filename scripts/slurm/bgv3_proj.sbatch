#!/bin/bash
#SBATCH --job-name=bgv3_proj
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=05:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/bgv3_proj_%j.log
# Node NODE of NNODES: waits for S5P_TTA (built by node 0 from the three Difix shifts) and the raw mix, then runs
# its 8 shards of the cross-target projection R = BGV3_<MIX>_raw, D = S5P_TTA -> BGV3_<MIX>_pAV.
#   for n in 0 1 2 3; do sbatch --export=ALL,MIX=ownT,LAM=15,NODE=$n,NNODES=4 work/slurm/bgv3_proj.sbatch; done
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${MIX:=ownT}" "${LAM:=15}" "${NODE:=0}" "${NNODES:=4}"
R=$V/submissions/BGV3_${MIX}_raw/renders; T=$V/submissions/S5P_TTA/renders; O=$V/submissions/BGV3_${MIX}_l${LAM}_pAV/renders
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
if [ "$NODE" = 0 ]; then
  until [ "$(cnt $V/submissions/S5P_difix)" -eq 2056 ] && [ "$(cnt $V/submissions/S5P_difix_s1)" -eq 2056 ] && [ "$(cnt $V/submissions/S5P_difix_s2)" -eq 2056 ]; do sleep 60; done
  echo "== S5P_TTA, $(date +%H:%M) =="
  $PY $W/tta_average.py $T $V/submissions/S5P_difix/renders $V/submissions/S5P_difix_s1/renders $V/submissions/S5P_difix_s2/renders --procs 90
  echo "S5P_TTA: $(cnt $T)/2056"
fi
until [ "$(cnt $T)" -eq 2056 ] && [ "$(cnt $R)" -eq 2056 ]; do sleep 60; done
echo "== proj node $NODE/$NNODES, R=$MIX D=S5P_TTA lam$LAM, $(date +%H:%M) =="
NS=$((NNODES*8))
for g in 0 1 2 3 4 5 6 7; do
  S=$((NODE*8+g))
  CUDA_VISIBLE_DEVICES=$g $PY $W/proj_multi.py tree $R $T $O --nets alex:1,vgg:0.5 --lam $LAM --steps 200 --batch 2 --shard $S/$NS > $V/jobs/bgv3proj_${MIX}_$S.log 2>&1 &
done; wait
echo "node $NODE done: $(cnt $O)/2056 $(date +%H:%M)"
echo BGV3_PROJ_NODE_DONE $NODE
