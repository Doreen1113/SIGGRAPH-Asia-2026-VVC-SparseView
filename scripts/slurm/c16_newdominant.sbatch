#!/bin/bash
#SBATCH --job-name=c16_newdom
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=05:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c16_newdom_%j.log
# The controlled C13-vs-C8 comparison isolated the new stride-5 x pseudo-view models: at only 3 of 9 weights in
# the S5P branch (~17% of the person source) they lifted FG-PSNR on every scene (+0.31 and FG-SSIM +0.0050 on
# 011). So stop diluting them. Here the person source is built from the SIX new models (wave22's S5PS/_s1/_w30
# plus wave23's _w40/_gate/_s2), with OLDW controlling how much of the old 6-seed S5P_ens is kept:
#   OLDW=0  -> new models only        NAME=C16
#   OLDW=6  -> 50/50 old vs new       NAME=C18
# Everything downstream is unchanged (NB3 mix, Difix 3-shift TTA, composite over BGX_tp15, feather 101), then the
# colour fix that test scores have now confirmed (011 taper scale 1.2, 007 gain 1.0) is layered on.
module load cuda/12.6 2>/dev/null
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
: "${OLDW:=0}"; : "${NAME:=C16}"
NEW="$S/M_S5PS/renders:1 $S/M_S5PS_s1/renders:1 $S/M_S5PS_w30/renders:1 $S/M_S5PS_w40/renders:1 $S/M_S5PS_gate/renders:1 $S/M_S5PS_s2/renders:1"
echo "== $NAME: waiting for the wave23 renders, $(date +%H:%M) =="
for m in M_S5PS_w40 M_S5PS_gate M_S5PS_s2; do
  until [ "$(cnt $S/$m)" -eq 2056 ]; do sleep 60; done
  echo "  $m ready $(date +%H:%M)"
done
P=$S/S5Pnew_${NAME}/renders
if [ "$OLDW" -gt 0 ]; then
  $PY $W/weighted_ensemble.py $P $S/S5P_ens/renders:$OLDW $NEW --procs 90
else
  $PY $W/weighted_ensemble.py $P $NEW --procs 90
fi
echo "person branch: $(cnt $P)/2056"
NB=$S/NB3_${NAME}/renders
$PY $W/weighted_ensemble.py $NB $S/SIX_ens/renders:4 $S/M_DFX_s0/renders:2 $S/M_DFXFG_L60K/renders:1 $S/M_DFXFG_L60K_s1/renders:1 $P:2 $S/M_S3_FGW40_L60K/renders:2 --procs 90
HB=$S/HB3_${NAME}/renders
$PY $W/tta_average.py $HB $P $NB --procs 90
echo "HB3_${NAME}: $(cnt $HB)/2056"
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=DFX_${NAME};; "320,288") vn=DFX_${NAME}_s1;; *) vn=DFX_${NAME}_s2;; esac
  echo "== difix $vn, $(date +%H:%M) =="
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $HB $S/$vn/renders ${s:+--shift $s} > $V/jobs/${NAME}_${vn}_$i.log 2>&1 & done; wait
done
TTA=$S/TTA_${NAME}/renders
$PY $W/tta_average.py $TTA $S/DFX_${NAME}/renders $S/DFX_${NAME}_s1/renders $S/DFX_${NAME}_s2/renders --procs 90
echo "TTA_${NAME}: $(cnt $TTA)/2056"
RAW=$S/${NAME}_raw/renders; i=0
for c in $CASES; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $HB $TTA 0.85 $S/BGX_tp15/renders 1.0 $RAW --cases $c --feather 101 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
  i=$((i+1))
done; wait
n=$(cnt $RAW); echo "-- ${NAME}_raw: $n/2056"
[ "$n" -eq 2056 ] || { echo "COMPOSITE INCOMPLETE"; exit 1; }
$PY $W/pack_submission.py $S/${NAME}_raw.zip $RAW && chmod 444 $S/${NAME}_raw.zip && echo "PACKED ${NAME}_raw $(date +%H:%M)"
# colour fix on top, at the scales the test set has confirmed
OUT=$S/${NAME}/renders; rm -rf $S/${NAME}/_x $OUT; mkdir -p $S/${NAME}/_x $OUT
$PY -c "import zipfile; zipfile.ZipFile('$S/${NAME}_raw.zip').extractall('$S/${NAME}/_x')"
mv $S/${NAME}/_x/sparseViewTrack/renders/* $OUT/; rm -rf $S/${NAME}/_x
rm -rf $OUT/011_0_seq0 $OUT/007_0_seq0
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.2 --shard $i/8 --tau 0.10 & done; wait
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 007_0_seq0 $W/priors/cc2_001_1.json gain 1.0 --shard $i/8 & done; wait
n2=$(cnt $OUT); echo "-- ${NAME}: $n2/2056"
[ "$n2" -eq 2056 ] && $PY $W/pack_submission.py $S/${NAME}.zip $OUT && chmod 444 $S/${NAME}.zip && echo "PACKED ${NAME} $(date +%H:%M)"
echo ${NAME}_CHAIN_DONE $(date +%H:%M)
