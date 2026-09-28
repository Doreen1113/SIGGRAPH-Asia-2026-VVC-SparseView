#!/bin/bash
#SBATCH --job-name=c41_half
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c41_half_%j.log
# C35 (100% temporal plate) bought FULL-SSIM +0.0057 but paid LPIPS +0.028 on every scene — enough to drop
# LPIPS from rank 2 to 3. SSIM only needed +0.0003 to flip. So dilute the plate: 50% and 25% blends keep the
# SSIM flip while giving back most of the LPIPS. Same person (C22 tree), same colour fix.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
$PY $W/weighted_ensemble.py $S/BGT_half/renders $S/BGT_new/renders:1 $S/BGX_new/renders:1 --procs 90
$PY $W/weighted_ensemble.py $S/BGT_quarter/renders $S/BGT_new/renders:1 $S/BGX_new/renders:3 --procs 90
for V2 in "C41_half:BGT_half" "C42_quarter:BGT_quarter"; do
  NAME=${V2%%:*}; BG=$S/${V2#*:}/renders
  [ "$(cnt $BG)" -eq 2056 ] || { echo "SKIP $NAME"; continue; }
  RAW=$S/${NAME}_raw/renders; i=0
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $S/HB3_C22/renders $S/TTA_C22/renders 0.85 $BG 1.0 $RAW --cases $c --feather 101 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
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
  rm -rf $RAW $OUT
done
echo C41_DONE $(date +%H:%M)
