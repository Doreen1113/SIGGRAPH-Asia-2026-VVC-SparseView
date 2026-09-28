#!/bin/bash
#SBATCH --job-name=c12_build
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/c12_build_%j.log
# C12 = C8c, plus the measured per-camera colour pattern applied to the cameras 004/006/009 SHARE with the
# validation rig (004: 02,17 | 006: 17,18 | 009: 17,18). Camera 17 alone carries 44% of the rig's squared
# colour offset, so 2 of 8 cameras still removes about half of each scene's colour error. Pattern only
# (zero-mean across the rig): the per-scene global term is unknown for these three scenes, so we do not guess it.
# 007/011 stay exactly as in C8c, so their per-scene metrics stay comparable and the result isolates this change.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
set -e
OUT=$S/C12_patcam/renders; rm -rf $S/C12_patcam; mkdir -p $OUT
$PY - <<'PYEOF'
import zipfile
z=zipfile.ZipFile('/work/doreen071/vvc/submissions/C8c.zip')
z.extractall('/work/doreen071/vvc/submissions/C12_patcam/_x')
PYEOF
mv $S/C12_patcam/_x/sparseViewTrack/renders/* $OUT/
rm -rf $S/C12_patcam/_x
echo "extracted: $(find $OUT -name '*.jpg' | wc -l)/2056"
for pair in "004_1_seq0:004:02 17" "006_1_seq0:006:17 18" "009_0_seq0:009:17 18"; do
  c=${pair%%:*}; rest=${pair#*:}; key=${rest%%:*}; cams=${rest#*:}
  for v in $cams; do rm -rf $OUT/$c/$v; done
  for i in 0 1 2 3 4 5 6 7; do
    $PY $W/camcolor2.py apply $S/C8c.zip $OUT $c $W/priors/pat_$key.json taper 1.0 --shard $i/8 --tau 0.10 &
  done; wait
done
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do echo "$c $(find $OUT/$c -name '*.jpg' | wc -l)"; done
n=$(find $OUT -name '*.jpg' | wc -l); echo "-- C12_patcam: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/C12_patcam.zip $OUT && chmod 444 $S/C12_patcam.zip && echo "PACKED C12_patcam $(date +%H:%M)"
echo C12_DONE $(date +%H:%M)
