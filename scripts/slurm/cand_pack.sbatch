#!/bin/bash
#SBATCH --job-name=cand_pack
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/cand_pack_%j.log
# Full (non-approximate) composite of the rebuilt HB3 person source over a background, then pack.
#   sbatch --export=ALL,BG=BGX_tp15,FEATHER=81,ALPHA=0.85,NAME=C1_hb3_tp15_f81 cand_pack.sbatch
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${BG:?}" "${FEATHER:=25}" "${ALPHA:=0.85}" "${NAME:?}"
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
until [ "$(cnt $S/HB3_TTA)" -eq 2056 ] && [ "$(cnt $S/$BG)" -eq 2056 ] && [ "$(cnt $S/HB3_ens)" -eq 2056 ]; do sleep 60; done
echo "== $NAME: HB3 a$ALPHA f$FEATHER over $BG, $(date +%H:%M) =="
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  for h in 0 1; do   # two processes per case, split by view parity via --cases only supports cases, so run one per case per GPU
    :
  done
  CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $S/HB3_ens/renders $S/HB3_TTA/renders $ALPHA $S/$BG/renders 1.0 $S/$NAME/renders --cases $c --feather $FEATHER > $V/jobs/cand_${NAME}_$c.log 2>&1 &
  i=$((i+1))
done; wait
n=$(cnt $S/$NAME); echo "-- $NAME: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/$NAME.zip $S/$NAME/renders && chmod 444 $S/$NAME.zip
echo CAND_PACK_DONE $NAME $(date +%H:%M)
