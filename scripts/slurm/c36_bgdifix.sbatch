#!/bin/bash
#SBATCH --job-name=c36_bgdifix
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c36_bgdifix_%j.log
# FULL-LPIPS is 0.0047 from rank 1 and the background is ~94% of the pixels behind it. The background has never
# been passed through Difix on the current tree (it only inherits Difix-ness via the projection target); the
# Sep-8 sweep found a background Difix strength near 0.45 optimal on a much older tree. Run 3-shift Difix TTA on
# the new projected background and blend it 50/50 with the un-Difixed version, under C22's person.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
SRC=$S/BGX_new/renders
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=DFX_BGnew;; "320,288") vn=DFX_BGnew_s1;; *) vn=DFX_BGnew_s2;; esac
  echo "== difix $vn, $(date +%H:%M) =="
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $W/difix_submission_h200.py $SRC $S/$vn/renders ${s:+--shift $s} > $V/jobs/c36_${vn}_$i.log 2>&1 & done; wait
done
$PY $W/tta_average.py $S/BGD_new/renders $S/DFX_BGnew/renders $S/DFX_BGnew_s1/renders $S/DFX_BGnew_s2/renders --procs 90
echo "BGD_new (difixed bg): $(cnt $S/BGD_new)/2056"
# 50/50 blend of projected background and its Difix TTA
$PY $W/tta_average.py $S/BGM_new/renders $S/BGX_new/renders $S/BGD_new/renders --procs 90
echo "BGM_new (blend): $(cnt $S/BGM_new)/2056"
for V2 in "C36_bgd50:BGM_new" "C37_bgd100:BGD_new"; do
  NAME=${V2%%:*}; BG=$S/${V2#*:}/renders
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
done
echo C36_DONE $(date +%H:%M)
