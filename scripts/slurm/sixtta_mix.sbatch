#!/bin/bash
#SBATCH --job-name=sixtta_mix
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/sixtta_mix_%j.log
# Build SIX_TTA (peer Difix target) once the three SIX Difix shifts exist, and the raw background mixes on test.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
until [ "$(cnt $S/SIX_difix)" -eq 2056 ] && [ "$(cnt $S/SIX_difix_s1)" -eq 2056 ] && [ "$(cnt $S/SIX_difix_s2)" -eq 2056 ]; do sleep 60; done
echo "== SIX_TTA, $(date +%H:%M) =="
$PY $W/tta_average.py $S/SIX_TTA/renders $S/SIX_difix/renders $S/SIX_difix_s1/renders $S/SIX_difix_s2/renders --procs 90
echo "SIX_TTA: $(cnt $S/SIX_TTA)/2056"
echo "== raw mixes, $(date +%H:%M) =="
[ "$(cnt $S/BGV3_ownT_raw)" -eq 2056 ] || echo "WARNING ownT raw incomplete: $(cnt $S/BGV3_ownT_raw)"
$PY $W/weighted_ensemble.py $S/BGV3_p4own_raw/renders $S/SIX_ens/renders:4 $S/BGV3_ownT_raw/renders:12 --procs 90
echo "BGV3_p4own_raw: $(cnt $S/BGV3_p4own_raw)/2056"
echo SIXTTA_MIX_DONE $(date +%H:%M)
