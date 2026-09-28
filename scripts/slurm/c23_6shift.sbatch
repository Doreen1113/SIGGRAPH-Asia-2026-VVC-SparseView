#!/bin/bash
#SBATCH --job-name=c23_6shift
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=05:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c23_6shift_%j.log
# Difix TTA currently averages 3 shifts. Averaging more independent shifts cuts the residual generative noise,
# and "less noise" is exactly the profile of the teams ahead of us (their LPIPS/SSIM lead, not sharpness).
# Adds 3 more shifts on top of C16's tree -> 6-shift TTA -> composite -> colour fix.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
HB=$S/HB3_C16/renders
until [ "$(cnt $HB)" -eq 2056 ]; do sleep 60; done
echo "== C16 HB3 ready, adding 3 more Difix shifts, $(date +%H:%M) =="
for s in "160,144" "480,240" "800,96"; do
  case "$s" in "160,144") vn=DFX_C16_s3;; "480,240") vn=DFX_C16_s4;; *) vn=DFX_C16_s5;; esac
  echo "== difix $vn, $(date +%H:%M) =="
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $HB $S/$vn/renders --shift $s > $V/jobs/c23_${vn}_$i.log 2>&1 & done; wait
done
for v in DFX_C16 DFX_C16_s1 DFX_C16_s2; do until [ "$(cnt $S/$v)" -eq 2056 ]; do sleep 60; done; done
TTA=$S/TTA6_C16/renders
$PY $W/tta_average.py $TTA $S/DFX_C16/renders $S/DFX_C16_s1/renders $S/DFX_C16_s2/renders $S/DFX_C16_s3/renders $S/DFX_C16_s4/renders $S/DFX_C16_s5/renders --procs 90
echo "TTA6: $(cnt $TTA)/2056"
NAME=C23_6shift; RAW=$S/${NAME}_raw/renders; i=0
for c in $CASES; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $HB $TTA 0.85 $S/BGX_tp15/renders 1.0 $RAW --cases $c --feather 101 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
  i=$((i+1))
done; wait
n=$(cnt $RAW); echo "-- ${NAME}_raw: $n/2056"
[ "$n" -eq 2056 ] || exit 1
$PY $W/pack_submission.py $S/${NAME}_raw.zip $RAW && chmod 444 $S/${NAME}_raw.zip
OUT=$S/${NAME}/renders; rm -rf $S/${NAME}/_x $OUT; mkdir -p $S/${NAME}/_x $OUT
$PY -c "import zipfile; zipfile.ZipFile('$S/${NAME}_raw.zip').extractall('$S/${NAME}/_x')"
mv $S/${NAME}/_x/sparseViewTrack/renders/* $OUT/; rm -rf $S/${NAME}/_x
rm -rf $OUT/011_0_seq0 $OUT/007_0_seq0
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.2 --shard $i/8 --tau 0.10 & done; wait
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 007_0_seq0 $W/priors/cc2_001_1.json gain 1.0 --shard $i/8 & done; wait
n2=$(cnt $OUT); echo "-- ${NAME}: $n2/2056"
[ "$n2" -eq 2056 ] && $PY $W/pack_submission.py $S/${NAME}.zip $OUT && chmod 444 $S/${NAME}.zip && echo "PACKED ${NAME} $(date +%H:%M)"
echo C23_DONE $(date +%H:%M)
