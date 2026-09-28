#!/bin/bash
#SBATCH --job-name=difix_HB3_difix_s2
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/difix_HB3_difix_s2_%j.log
module load cuda/12.6 2>/dev/null || true
V=/work/doreen071/vvc; PY=/home/doreen071/.conda/envs/vvc/bin/python
until grep -q HB3_ENS_DONE $V/jobs/hb3_build_*.log 2>/dev/null; do sleep 60; done
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $V/work/difix_submission_h200.py $V/submissions/HB3_ens/renders $V/submissions/HB3_difix_s2/renders --shift 640,192 > $V/jobs/difix_HB3_difix_s2_shard$i.log 2>&1 &
done; wait
echo "HB3_difix_s2: $(find $V/submissions/HB3_difix_s2/renders -name '*.jpg' | wc -l)/2056"; echo DIFIX_HB3_difix_s2_DONE $(date +%H:%M)
