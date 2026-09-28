#!/bin/bash
#SBATCH --job-name=c8c_build
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=00:40:00
#SBATCH --output=/work/doreen071/vvc/jobs/c8c_build_%j.log
# C8c = C8_cc2 with 011's taper scale bumped 1.0 -> 1.2 (val sweep: PSNR +0.18, LPIPS -0.0004, SSIM ~flat).
# 007/004/006/009 are byte-identical to C8_cc2.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
OUT=$S/C8c/renders; mkdir -p $OUT
for i in 0 1 2 3 4 5 6 7; do
  $PY $W/camcolor2.py apply $S/C0_hb3_nbx15_f101.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.2 --shard $i/8 --tau 0.10 &
done; wait
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0; do
  rm -rf $OUT/$c; mkdir -p /tmp/xtr_$c
  $PY -c "
import zipfile
z=zipfile.ZipFile('$S/C8_cc2.zip')
for n in z.namelist():
    if '/renders/$c/' in n: z.extract(n,'/tmp/xtr_$c')
"
  mv /tmp/xtr_$c/sparseViewTrack/renders/$c $OUT/$c
  rm -rf /tmp/xtr_$c
done
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do echo "$c $(find $OUT/$c -name '*.jpg' | wc -l)"; done
n=$(find $OUT -name '*.jpg' | wc -l); echo "-- C8c: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/C8c.zip $OUT && chmod 444 $S/C8c.zip && echo "PACKED C8c $(date +%H:%M)"
echo C8C_DONE $(date +%H:%M)
