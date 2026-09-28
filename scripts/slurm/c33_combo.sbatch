#!/bin/bash
#SBATCH --job-name=c33_combo
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c33_combo_%j.log
# Every candidate so far has improved EITHER the person OR the background, never both: C22 (best person, from the
# new-model-dominant tree) still composites over the OLD BGX_tp15, and C31's clean background was built on the OLD
# person tree. Both background upgrades are now finished, so combine them with C22's person tree:
#   C33 = C22 person + temporally averaged background (targets FULL-SSIM +0.0003 and FULL-LPIPS -0.0047)
#   C34 = C22 person + BGX_new (background re-projected from the new-model render)
#   C35 = C22 person + temporal averaging applied on top of BGX_new (both background gains stacked)
# The three near-free rank flips we still have not banked are FG-SSIM +0.0002, FULL-SSIM +0.0003, FULL-LPIPS -0.0047.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
# stack temporal averaging on the new projected background
i=0
for c in $CASES; do
  CUDA_VISIBLE_DEVICES=$i $PY $W/temporal_bg_safe.py $S/BGX_new/renders $S/HB3_C22/renders $S/BGT_new/renders --case $c > $V/jobs/tbgnew_$c.log 2>&1 &
  i=$((i+1))
done; wait
echo "BGT_new: $(cnt $S/BGT_new)/2056"
build() {  # $1 name  $2 background root
  NAME=$1; BG=$2
  [ "$(cnt $BG)" -eq 2056 ] || { echo "SKIP $NAME (background incomplete)"; return; }
  RAW=$S/${NAME}_raw/renders; i=0
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $S/HB3_C22/renders $S/TTA_C22/renders 0.85 $BG 1.0 $RAW --cases $c --feather 101 > $V/jobs/cand_${NAME}_$c.log 2>&1 &
    i=$((i+1))
  done; wait
  n=$(cnt $RAW); echo "-- ${NAME}_raw: $n/2056"; [ "$n" -eq 2056 ] || return
  $PY $W/pack_submission.py $S/${NAME}_raw.zip $RAW && chmod 444 $S/${NAME}_raw.zip
  OUT=$S/${NAME}/renders; rm -rf $S/${NAME}/_x $OUT; mkdir -p $S/${NAME}/_x $OUT
  $PY -c "import zipfile; zipfile.ZipFile('$S/${NAME}_raw.zip').extractall('$S/${NAME}/_x')"
  mv $S/${NAME}/_x/sparseViewTrack/renders/* $OUT/; rm -rf $S/${NAME}/_x
  rm -rf $OUT/011_0_seq0 $OUT/007_0_seq0
  for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.2 --shard $i/8 --tau 0.10 & done; wait
  for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/${NAME}_raw.zip $OUT 007_0_seq0 $W/priors/cc2_001_1.json gain 1.0 --shard $i/8 & done; wait
  n2=$(cnt $OUT); echo "-- ${NAME}: $n2/2056"
  [ "$n2" -eq 2056 ] && $PY $W/pack_submission.py $S/${NAME}.zip $OUT && chmod 444 $S/${NAME}.zip && echo "PACKED ${NAME} $(date +%H:%M)"
}
build C33_p22_tbg  $S/BGT_safe/renders
build C34_p22_bgnew $S/BGX_new/renders
build C35_p22_both  $S/BGT_new/renders
echo C33_COMBO_DONE $(date +%H:%M)
