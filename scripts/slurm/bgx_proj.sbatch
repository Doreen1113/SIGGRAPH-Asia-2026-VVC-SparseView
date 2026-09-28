#!/bin/bash
#SBATCH --job-name=bgx_proj
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/bgx_proj_%j.log
# One node of a cross-target background projection: R = <RAW> mix, D = SIX_TTA (peer Difix), lam LAM -> <OUT>.
#   for n in 0..7: sbatch --export=ALL,RAW=BGV3_ownT_raw,OUT=BGX_tp15,LAM=15,NODE=$n,NNODES=8 bgx_proj.sbatch
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${RAW:?}" "${OUT:?}" "${LAM:=15}" "${NODE:?}" "${NNODES:=8}"
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
until [ "$(cnt $S/SIX_TTA)" -eq 2056 ] && [ "$(cnt $S/$RAW)" -eq 2056 ]; do sleep 60; done
echo "== $OUT node $NODE/$NNODES start $(date +%H:%M) =="
NS=$((NNODES*8))
for g in 0 1 2 3 4 5 6 7; do
  SH=$((NODE*8+g))
  CUDA_VISIBLE_DEVICES=$g $PY $W/proj_multi.py tree $S/$RAW/renders $S/SIX_TTA/renders $S/$OUT/renders --nets alex:1,vgg:0.5 --lam $LAM --steps 200 --batch 2 --shard $SH/$NS > $V/jobs/bgx_${OUT}_$SH.log 2>&1 &
done; wait
echo "$OUT node $NODE done, total now $(cnt $S/$OUT)/2056 $(date +%H:%M)"
echo BGX_PROJ_NODE_DONE
