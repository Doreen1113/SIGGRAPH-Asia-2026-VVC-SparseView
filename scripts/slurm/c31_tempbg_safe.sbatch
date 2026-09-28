#!/bin/bash
#SBATCH --job-name=c31_tbgsafe
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c31_tbgsafe_%j.log
# C27 (temporal background plate) moved FULL-SSIM +0.0057 — ten times the threshold we need — but destroyed the
# foreground (-3.0 FG-PSNR) because the plate was built with its own person mask and then composited under a
# DIFFERENT mask, so person ghosts baked into the plate showed through near the silhouette.
# Fix: only trust the plate where the person NEVER goes. Build the union of every frame's person mask for a
# view, dilate it hard, and keep the original per-frame background inside that union. Far background (which is
# what FULL-SSIM/LPIPS are made of) gets the clean plate; anything the evaluator crops for FG is untouched.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
BGOUT=$S/BGT_safe/renders
i=0
for c in $CASES; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/temporal_bg_safe.py $S/BGX_tp15/renders $S/HB3_ens/renders $BGOUT --case $c > $V/jobs/tbgsafe_$c.log 2>&1 &
  i=$((i+1))
done; wait
echo "BGT_safe: $(cnt $BGOUT)/2056"
[ "$(cnt $BGOUT)" -eq 2056 ] || { echo INCOMPLETE; exit 1; }
NAME=C31_tbgsafe; RAW=$S/${NAME}_raw/renders; i=0
for c in $CASES; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $S/HB3_ens/renders $S/HB3_TTA/renders 0.85 $BGOUT 1.0 $RAW --cases $c --feather 101 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
  i=$((i+1))
done; wait
n=$(cnt $RAW); echo "-- ${NAME}_raw: $n/2056"; [ "$n" -eq 2056 ] || exit 1
$PY $W/pack_submission.py $S/${NAME}_raw.zip $RAW && chmod 444 $S/${NAME}_raw.zip
OUT=$S/${NAME}/renders; rm -rf $S/${NAME}/_x $OUT; mkdir -p $S/${NAME}/_x $OUT
$PY -c "import zipfile; zipfile.ZipFile('$S/${NAME}_raw.zip').extractall('$S/${NAME}/_x')"
mv $S/${NAME}/_x/sparseViewTrack/renders/* $OUT/; rm -rf $S/${NAME}/_x
rm -rf $OUT/011_0_seq0 $OUT/007_0_seq0
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.2 --shard $i/8 --tau 0.10 & done; wait
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 007_0_seq0 $W/priors/cc2_001_1.json gain 1.0 --shard $i/8 & done; wait
n2=$(cnt $OUT); echo "-- ${NAME}: $n2/2056"
[ "$n2" -eq 2056 ] && $PY $W/pack_submission.py $S/${NAME}.zip $OUT && chmod 444 $S/${NAME}.zip && echo "PACKED ${NAME} $(date +%H:%M)"
echo C31_DONE $(date +%H:%M)
