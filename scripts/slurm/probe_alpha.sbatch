#!/bin/bash
#SBATCH --job-name=probe_alpha
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/probe_alpha_%j.log
# Probe material: alpha 0.95 / 1.00 (stronger Difix on the person) for the four scenes where only 0.85 was ever scored.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
g=0
for A in 0.95 1.00; do
  T=$(printf "a%03d" $(echo "$A*100/1" | bc))
  for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0; do
    CUDA_VISIBLE_DEVICES=$g $PY $W/composite2.py tree $S/HB3_ens/renders $S/HB3_TTA/renders $A $S/BGX_tp15/renders 1.0 $S/PROBE_$T/renders --cases $c --feather 101 > $V/jobs/probe_${T}_$c.log 2>&1 &
    g=$((g+1))
  done
done; wait
for T in a095 a100; do for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0; do echo "$T $c $(find $S/PROBE_$T/renders/$c -name '*.jpg' | wc -l)"; done; done
echo PROBE_ALPHA_DONE $(date +%H:%M)
