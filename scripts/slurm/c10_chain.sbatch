#!/bin/bash
#SBATCH --job-name=c10_chain
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c10_chain_%j.log
# Waits for C9 (wave22 9-seed person ensemble) to pack, then layers the validated colour correction
# (taper on 011, gain on 007) on top -> C10. Built but NOT auto-submitted: C9's model quality is unproven
# (no GT for 011/007 test scenes), so this only becomes a candidate after a human decision, not blind upload.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
until [ -f $S/C9_hb3v4_tp15_f101.zip ]; do sleep 60; done
echo "== C9 ready, layering colour fix, $(date +%H:%M) =="
OUT=$S/C10_cc/renders; mkdir -p $OUT
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/C9_hb3v4_tp15_f101.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.0 --shard $i/8 --tau 0.10 & done; wait
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/C9_hb3v4_tp15_f101.zip $OUT 007_0_seq0 $W/priors/cc2_001_1.json gain 1.0 --shard $i/8 & done; wait
for c in 004_1_seq0 006_1_seq0 009_0_seq0; do
  rm -rf $OUT/$c; mkdir -p $OUT/$c
  Z=$S/C9_hb3v4_tp15_f101.zip
  $PY -c "
import zipfile
z=zipfile.ZipFile('$Z')
for n in z.namelist():
    if '/renders/$c/' in n: z.extract(n, '$OUT/../')
"
done
mv $OUT/../sparseViewTrack/renders/004_1_seq0 $OUT/004_1_seq0 2>/dev/null
mv $OUT/../sparseViewTrack/renders/006_1_seq0 $OUT/006_1_seq0 2>/dev/null
mv $OUT/../sparseViewTrack/renders/009_0_seq0 $OUT/009_0_seq0 2>/dev/null
rm -rf $S/C10_cc/sparseViewTrack
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do echo "$c $(find $OUT/$c -name '*.jpg' | wc -l)"; done
n=$(find $OUT -name '*.jpg' | wc -l); echo "-- C10_cc: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/C10_cc.zip $OUT && chmod 444 $S/C10_cc.zip && echo "PACKED C10_cc $(date +%H:%M) -- NOT auto-submitted, awaiting decision"
echo C10_CHAIN_DONE $(date +%H:%M)
