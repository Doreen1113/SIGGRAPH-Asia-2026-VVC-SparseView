#!/bin/bash
#SBATCH --job-name=c13_build
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c13_build_%j.log
# C13 = C9 (new 9-seed person ensemble) with the validated colour fix layered on 007/011 (same recipe as C8c).
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
set -e
rm -rf $S/C13/_x $S/C13/renders; mkdir -p $S/C13/_x $S/C13/renders
$PY -c "
import zipfile
z=zipfile.ZipFile('$S/C9_hb3v4_tp15_f101.zip')
z.extractall('$S/C13/_x')
"
mv $S/C13/_x/sparseViewTrack/renders/* $S/C13/renders/
rm -rf $S/C13/_x
OUT=$S/C13/renders
rm -rf $OUT/011_0_seq0 $OUT/007_0_seq0
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/C9_hb3v4_tp15_f101.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.2 --shard $i/8 --tau 0.10 & done; wait
for i in 0 1 2 3 4 5 6 7; do $PY $W/camcolor2.py apply $S/C9_hb3v4_tp15_f101.zip $OUT 007_0_seq0 $W/priors/cc2_001_1.json gain 1.0 --shard $i/8 & done; wait
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do echo "$c $(find $OUT/$c -name '*.jpg' | wc -l)"; done
n=$(find $OUT -name '*.jpg' | wc -l); echo "-- C13: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/C13.zip $OUT && chmod 444 $S/C13.zip && echo "PACKED C13 $(date +%H:%M)"
echo C13_DONE $(date +%H:%M)
