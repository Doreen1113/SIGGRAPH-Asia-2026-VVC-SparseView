#!/bin/bash
#SBATCH --job-name=cand_011a
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/cand_011alpha_%j.log
# 011_0 is the only scene where MORE Difix helped all six (HB3 p085 > p070 on every metric on 011). Test alpha 0.95
# and 1.0 on 011 only, spliced into C1 (rebuilt HB3 + BGX_tp15 + feather 101) for the other four scenes.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
g=0
for A in 0.95 1.00; do
  T=$(printf "a%03d" $(echo "$A*100/1" | bc)); OUT=$S/C1_011${T}
  CUDA_VISIBLE_DEVICES=$g $PY $W/composite2.py tree $S/HB3_ens/renders $S/HB3_TTA/renders $A $S/BGX_tp15/renders 1.0 $OUT/renders --cases 011_0_seq0 --feather 101 > $V/jobs/cand011_$T.log 2>&1 &
  g=$((g+1))
done; wait
for A in 0.95 1.00; do
  T=$(printf "a%03d" $(echo "$A*100/1" | bc)); OUT=$S/C1_011${T}
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0; do ln -sfn $S/C1_hb3_tp15_f101/renders/$c $OUT/renders/$c; done
  n=$(cnt $OUT/renders); echo "-- C1_011$T: $n/2056"
  [ "$n" -eq 2056 ] && $PY $W/pack_submission.py $S/C1_011$T.zip $OUT/renders && chmod 444 $S/C1_011$T.zip && echo "PACKED C1_011$T $(date +%H:%M)"
done
echo CAND_011A_DONE $(date +%H:%M)
