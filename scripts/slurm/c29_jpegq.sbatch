#!/bin/bash
#SBATCH --job-name=c29_jpegq
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c29_jpegq_%j.log
# Final-JPEG-quality variants. Measured on val with single compression: q90 gives FULL-SSIM +0.0013 and
# FULL-LPIPS +0.0119. Our FULL-SSIM is 0.0005 from a rank while our FULL-LPIPS has 0.018 of slack before the
# team below us — so trading LPIPS headroom for SSIM is a rank in our favour. q92 is the conservative point,
# q90 the aggressive one; per-scene scores will say which (or which mix) actually lands.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
CASES="004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0"
for Q in 92 90; do
  NAME=C29_q$Q; RAW=$S/${NAME}_raw/renders; i=0
  for c in $CASES; do
    CUDA_VISIBLE_DEVICES=$i $PY $W/composite2.py tree $S/HB3_ens/renders $S/HB3_TTA/renders 0.85 $S/BGX_tp15/renders 1.0 $RAW --cases $c --feather 101 --q $Q > $V/jobs/cand_${NAME}_$c.log 2>&1 &
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
echo C29_DONE $(date +%H:%M)
