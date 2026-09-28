#!/bin/bash
#SBATCH --job-name=cand_multi
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/cand_multi_%j.log
# Build all candidate packages sequentially on one node (each: full composite of rebuilt HB3 over a background).
# SPECS: "NAME BG FEATHER ALPHA" lines; each waits for its background tree, HB3_TTA and the faithfulness check.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
until [ "$(cnt $S/HB3_TTA)" -eq 2056 ] && [ "$(cnt $S/HB3_ens)" -eq 2056 ]; do sleep 60; done
until grep -q HB3R_CHECK_DONE $V/jobs/hb3r_check_389899.log 2>/dev/null; do sleep 30; done
grep "rebuilt vs" $V/jobs/hb3r_check_389899.log
SPECS="C0_hb3_nbx15_f101 NBX15_pAV 101 0.85
C1_hb3_tp15_f101 BGX_tp15 101 0.85
C2_hb3_bp4_f101 BGX_bp4 101 0.85
C3_hb3_tp15_f131 BGX_tp15 131 0.85
C4_hb3_bp4_f81 BGX_bp4 81 0.85"
echo "$SPECS" | while read NAME BG FEATHER ALPHA; do
  [ -z "$NAME" ] && continue
  until [ "$(cnt $S/$BG)" -eq 2056 ]; do sleep 60; done
  echo "== $NAME: HB3 a$ALPHA f$FEATHER over $BG, $(date +%H:%M) =="
  i=0
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
    CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $S/HB3_ens/renders $S/HB3_TTA/renders $ALPHA $S/$BG/renders 1.0 $S/$NAME/renders --cases $c --feather $FEATHER > $V/jobs/cand_${NAME}_$c.log 2>&1 &
    i=$((i+1))
  done; wait
  n=$(cnt $S/$NAME); echo "-- $NAME: $n/2056 $(date +%H:%M)"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/$NAME.zip $S/$NAME/renders && chmod 444 $S/$NAME.zip && echo "PACKED $NAME $(date +%H:%M)"
done
echo CAND_MULTI_DONE $(date +%H:%M)
