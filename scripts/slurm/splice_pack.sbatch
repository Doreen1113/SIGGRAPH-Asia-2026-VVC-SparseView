#!/bin/bash
#SBATCH --job-name=splice_pack
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/work/doreen071/vvc/jobs/splice_pack_%j.log
V=/work/doreen071/vvc; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
for T in a095 a100; do
  OUT=$S/C1_011$T/renders
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0; do rm -f $OUT/$c; cp -al $S/C1_hb3_tp15_f101/renders/$c $OUT/$c; done
  n=$(find $OUT -name '*.jpg' | wc -l); echo "-- C1_011$T: $n/2056"
  [ "$n" -eq 2056 ] && $PY $V/work/pack_submission.py $S/C1_011$T.zip $OUT && chmod 444 $S/C1_011$T.zip && echo "PACKED C1_011$T $(date +%H:%M)"
done
echo SPLICE_DONE
