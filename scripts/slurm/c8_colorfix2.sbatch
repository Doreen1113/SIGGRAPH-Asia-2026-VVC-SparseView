#!/bin/bash
#SBATCH --job-name=c8_cc2
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/c8_cc2_%j.log
# C8 = S1_splice with the two rig-matched scenes colour-corrected in the form their own validation scene prefers:
#   011_0 <- C0 renders + taper(tau=0.10)   [012_0: all six up, dark-window damage 1/3 of the additive form]
#   007_0 <- C1_011a100 renders + gain      [001_1: all six up; the additive form LOSES SSIM there, as on test]
# Other scenes are byte-identical to S1_splice (004/006 from C3, 009 from HB3_p085_nbx15_f101s4).
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
set -e
$PY $W/camcolor2.py fit $V/fixtest/012_0 '{D}/ps_lev/{tag}_f101.png'   $W/priors/cc2_012_0.json
$PY $W/camcolor2.py fit $V/fixtest/001_1 '{D}/ps_cur/{tag}_c_x_tp15.png' $W/priors/cc2_001_1.json
OUT=$S/C8_cc2/renders; mkdir -p $OUT
for i in 0 1 2 3 4 5 6 7; do
  $PY $W/camcolor2.py apply $S/C0_hb3_nbx15_f101.zip $OUT 011_0_seq0 $W/priors/cc2_012_0.json taper 1.0 --shard $i/8 --tau 0.10 &
done; wait
for i in 0 1 2 3 4 5 6 7; do
  $PY $W/camcolor2.py apply $S/C1_011a100.zip $OUT 007_0_seq0 $W/priors/cc2_001_1.json gain 1.0 --shard $i/8 &
done; wait
for c in 004_1_seq0 006_1_seq0; do rm -rf $OUT/$c; cp -al $S/C3_hb3_tp15_f131/renders/$c $OUT/$c; done
rm -rf $OUT/009_0_seq0; cp -al $S/HB3_p085_nbx15_f101s4/renders/009_0_seq0 $OUT/009_0_seq0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do echo "$c $(find $OUT/$c -name '*.jpg' | wc -l)"; done
n=$(find $OUT -name '*.jpg' | wc -l); echo "-- C8_cc2: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/C8_cc2.zip $OUT && chmod 444 $S/C8_cc2.zip && echo "PACKED C8_cc2 $(date +%H:%M)"
echo C8_CC2_DONE $(date +%H:%M)
