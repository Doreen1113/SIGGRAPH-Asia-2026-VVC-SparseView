#!/bin/bash
#SBATCH --job-name=c26_difix2
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c26_difix2_%j.log
# Second Difix pass over the already-Difix'd TTA output. Difix is a restoration model; a second pass either
# removes the residual generative noise (which is what separates the teams above us on LPIPS/SSIM) or
# over-smooths and costs SSIM. Cheap to settle, and it runs on the EXISTING HB3 tree so it needs no new models.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
SRC=$S/HB3_TTA/renders
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=D2_HB3;; "320,288") vn=D2_HB3_s1;; *) vn=D2_HB3_s2;; esac
  echo "== second-pass difix $vn, $(date +%H:%M) =="
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $SRC $S/$vn/renders ${s:+--shift $s} > $V/jobs/c26_${vn}_$i.log 2>&1 & done; wait
done
TTA2=$S/HB3_TTA2/renders
$PY $W/tta_average.py $TTA2 $S/D2_HB3/renders $S/D2_HB3_s1/renders $S/D2_HB3_s2/renders --procs 90
echo "HB3_TTA2: $(cnt $TTA2)/2056"
# two variants: person = second-pass output at the usual 0.85, and a milder 0.60 in case the second pass oversmooths
for A in 0.85 0.60; do
  T=$(printf "a%03d" $(echo "$A*100/1" | bc)); NAME=C26_$T
  RAW=$S/${NAME}_raw/renders; i=0
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $S/HB3_ens/renders $TTA2 $A $S/BGX_tp15/renders 1.0 $RAW --cases $c --feather 101 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
    i=$((i+1))
  done; wait
  n=$(cnt $RAW); echo "-- ${NAME}_raw: $n/2056"; [ "$n" -eq 2056 ] || continue
  $PY $W/pack_submission.py $S/${NAME}_raw.zip $RAW && chmod 444 $S/${NAME}_raw.zip
  OUT=$S/${NAME}/renders; rm -rf $S/${NAME}/_x $OUT; mkdir -p $S/${NAME}/_x $OUT
  $PY -c "import zipfile; zipfile.ZipFile('$S/${NAME}_raw.zip').extractall('$S/${NAME}/_x')"
  mv $S/${NAME}/_x/sparseViewTrack/renders/* $OUT/; rm -rf $S/${NAME}/_x
  rm -rf $OUT/011_0_seq0 $OUT/007_0_seq0
  for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.2 --shard $i/8 --tau 0.10 & done; wait
  for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 007_0_seq0 $W/priors/cc2_001_1.json gain 1.0 --shard $i/8 & done; wait
  n2=$(cnt $OUT); echo "-- ${NAME}: $n2/2056"
  [ "$n2" -eq 2056 ] && $PY $W/pack_submission.py $S/${NAME}.zip $OUT && chmod 444 $S/${NAME}.zip && echo "PACKED ${NAME} $(date +%H:%M)"
done
echo C26_DONE $(date +%H:%M)
