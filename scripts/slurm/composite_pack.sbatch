#!/bin/bash
#SBATCH --job-name=composite
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:5
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/composite_%j.log
# Usage: sbatch --export=ALL,FIX=<difix_root>,OUT=<name>,PA=0.60,BA=0.55 composite_pack.sbatch
PY=/home/doreen071/.conda/envs/vvc/bin/python
SRC=/work/doreen071/vvc/submissions/SIX_ens/renders
DST=/work/doreen071/vvc/submissions/$OUT/renders
export OMP_NUM_THREADS=6
i=0
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  CUDA_VISIBLE_DEVICES=$i $PY /work/doreen071/vvc/work/person_bg_composite.py $SRC $FIX $DST --pa $PA --ba $BA --cases $c \
    > /work/doreen071/vvc/jobs/composite_${OUT}_$c.log 2>&1 &
  i=$((i+1))
done
wait
for c in $(ls $SRC); do echo "$c: $(find $DST/$c -name '*.jpg' | wc -l) / $(find $SRC/$c -name '*.jpg' | wc -l)"; done
$PY /work/doreen071/vvc/work/pack_submission.py /work/doreen071/vvc/submissions/$OUT.zip $DST
echo COMPOSITE_${OUT}_DONE
