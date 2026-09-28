#!/bin/bash
#SBATCH --job-name=hb3r_build
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/hb3r_build_%j.log
# Rebuild the HB3 person source (deleted in the quota clean) now that SIX_ens is back:
#   NB3 = SIX:4 DFX:2 DFXFG:1 DFXFG_s1:1 S5P:2 S3:2 ; HB3_ens = mean(S6E_ens(=S5P_ens), NB3_ens)
# then Difix (SHIFT env picks one shift; first copy builds the ensembles, the others wait for HB3_ens).
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SHIFT="${SHIFT:-}"; SHIFT="${SHIFT//x/,}"
case "$SHIFT" in "") VN=HB3_difix;; "320,288") VN=HB3_difix_s1;; *) VN=HB3_difix_s2;; esac
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
S=$V/submissions
if [ -z "$SHIFT" ]; then
  echo "== NB3_ens, $(date +%H:%M) =="
  [ "$(cnt $S/NB3_ens)" -eq 2056 ] || $PY $W/weighted_ensemble.py $S/NB3_ens/renders $S/SIX_ens/renders:4 $S/M_DFX_s0/renders:2 \
    $S/M_DFXFG_L60K/renders:1 $S/M_DFXFG_L60K_s1/renders:1 $S/S5P_ens/renders:2 $S/M_S3_FGW40_L60K/renders:2 --procs 90
  echo "NB3_ens: $(cnt $S/NB3_ens)/2056"
  echo "== HB3_ens, $(date +%H:%M) =="
  [ "$(cnt $S/HB3_ens)" -eq 2056 ] || $PY $W/tta_average.py $S/HB3_ens/renders $S/S5P_ens/renders $S/NB3_ens/renders --procs 90
  echo "HB3_ens: $(cnt $S/HB3_ens)/2056"
fi
until [ "$(cnt $S/HB3_ens)" -eq 2056 ]; do sleep 30; done
echo "== difix $VN, $(date +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $S/HB3_ens/renders $S/$VN/renders ${SHIFT:+--shift $SHIFT} > $V/jobs/hb3r_${VN}_$i.log 2>&1 &
done; wait
echo "$VN: $(cnt $S/$VN)/2056"
echo HB3R_DONE $VN $(date +%H:%M)
