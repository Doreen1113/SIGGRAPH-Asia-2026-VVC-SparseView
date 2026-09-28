#!/bin/bash
#SBATCH --job-name=fge_ens
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fge_ens_%j.log
# Person source = equal mean of the 4 FG-specialist members we have rendered on the test scenes
# (val fe0 analogue: pure FG ensemble, no peer members).
V=/work/doreen071/vvc; PY=/home/doreen071/.conda/envs/vvc/bin/python
$PY $V/work/tta_average.py $V/submissions/FGE_ens/renders \
  $V/submissions/M_FGW40_L60K/renders $V/submissions/M_FGW20_s0/renders $V/submissions/M_FGW80_s0/renders $V/submissions/M_LONG60K_s0/renders --procs 90
echo "FGE_ens: $(find $V/submissions/FGE_ens -name '*.jpg' | wc -l)/2056"
echo FGE_ENS_DONE $(date +%H:%M)
