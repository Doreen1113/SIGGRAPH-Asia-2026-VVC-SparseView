#!/bin/bash
#SBATCH --job-name=bgv3_build
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/bgv3_build_%j.log
# Test-set build of background v3 ingredients: S5P_ens (6-seed mean), the own-model raw mix (MIX env, weights as in
# val_bgv3), and ONE Difix shift of S5P_ens (SHIFT env: "", "320,288" or "640,192") -> S5P_difix[_s1|_s2].
# Run three copies (one per shift) on three nodes; bgv3_proj.sbatch waits for all three.
#   sbatch --export=ALL,SHIFT=320x288 work/slurm/bgv3_build.sbatch   (shift given with x, not comma)
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${MIX:=ownT}"; SHIFT="${SHIFT:-}"; SHIFT="${SHIFT//x/,}"   # pass as 320x288 (commas break --export)
case "$SHIFT" in "") VN=S5P_difix;; "320,288") VN=S5P_difix_s1;; *) VN=S5P_difix_s2;; esac
S5=$V/submissions/S5P_ens/renders
if [ "$(find $S5 -name '*.jpg' 2>/dev/null | wc -l)" -ne 2056 ]; then
  echo "== S5P_ens, $(date +%H:%M) =="
  $PY $W/tta_average.py $S5 $V/submissions/M_S5_FGW40_L60K/renders $V/submissions/M_S5_FGW40_L60K_s1/renders $V/submissions/M_S5_FGW40_L60K_s2/renders \
    $V/submissions/M_S5_FGW40_L60K_s3/renders $V/submissions/M_S5_FGW40_L60K_s4/renders $V/submissions/M_S5_FGW40_L60K_s5/renders --procs 90
fi
echo "S5P_ens: $(find $S5 -name '*.jpg' | wc -l)/2056"
T10=$V/submissions/T10_ens/renders
if [ -z "$SHIFT" ]; then   # only the first copy builds the mixes
  [ "$(find $T10 -name '*.jpg' 2>/dev/null | wc -l)" -eq 2056 ] || $PY $W/tta_average.py $T10 $V/submissions/M_FGW40_L60K/renders $V/submissions/M_FGW40_L60K_s1/renders \
    $V/submissions/M_FGW40_L60K_s2/renders $V/submissions/M_FGW40_L60K_s3/renders $V/submissions/M_FGW40_L60K_s4/renders \
    $V/submissions/M_FGW20_s0/renders $V/submissions/M_FGW80_s0/renders $V/submissions/M_LONG60K_s0/renders $V/submissions/M_S3_FGW40_L60K/renders --procs 90
  echo "T10_ens: $(find $T10 -name '*.jpg' | wc -l)/2056"
  M=$V/submissions/M_DFX_s0/renders; G=$V/submissions/M_DFXFG_L60K/renders; G1=$V/submissions/M_DFXFG_L60K_s1/renders; N=$V/submissions/M_NBD_S5_FG/renders
  case "$MIX" in
    ownT)   ARGS="$M:2 $G:1 $G1:1 $S5:2 $N:2 $T10:4";;
    own6)   ARGS="$M:2 $G:1 $G1:1 $S5:2 $N:2 $T10:2";;
    own6nn) ARGS="$M:2 $G:1 $G1:1 $S5:2 $T10:2";;
    *) echo "unknown MIX $MIX"; exit 1;;
  esac
  echo "== mix $MIX, $(date +%H:%M) =="
  $PY $W/weighted_ensemble.py $V/submissions/BGV3_${MIX}_raw/renders $ARGS --procs 90
  echo "BGV3_${MIX}_raw: $(find $V/submissions/BGV3_${MIX}_raw -name '*.jpg' | wc -l)/2056"
fi
echo "== difix $VN, $(date +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $S5 $V/submissions/$VN/renders ${SHIFT:+--shift $SHIFT} > $V/jobs/bgv3_${VN}_$i.log 2>&1 &
done; wait
echo "$VN: $(find $V/submissions/$VN -name '*.jpg' | wc -l)/2056"
echo BGV3_BUILD_DONE $VN $(date +%H:%M)
