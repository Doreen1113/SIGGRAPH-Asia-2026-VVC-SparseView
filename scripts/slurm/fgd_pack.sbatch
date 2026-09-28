#!/bin/bash
#SBATCH --job-name=fgd_pack
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fgd_pack_%j.log
# Our person mask covers 0.0526 of the frame; the evaluator's foreground is 0.0596. About an eighth of the region
# it scores never receives our FG-specialist content. Dilate the mask by 12 px and measure what that ring is worth.
V=/work/doreen071/vvc; W=$V/work; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py tree $V/submissions/FGE6_ens/renders $V/submissions/FGE6_TTA/renders 0.85 \
    $V/submissions/SIX_pAVt15/renders 1.0 $V/submissions/FGD_p085_bpAVt15/renders --cases $c --dilate 12 > $V/jobs/fgdcomp_$c.log 2>&1 &
  i=$((i+1))
done
wait
OUT=FGD_p085_bpAVt15; n=$(find $V/submissions/$OUT -name '*.jpg' 2>/dev/null | wc -l)
echo "-- $OUT: $n/2056"
[ "$n" -eq 2056 ] && $PY $W/pack_submission.py $V/submissions/$OUT.zip $V/submissions/$OUT/renders && chmod 444 $V/submissions/$OUT.zip
echo FGD_PACK_DONE $(date +%H:%M)
