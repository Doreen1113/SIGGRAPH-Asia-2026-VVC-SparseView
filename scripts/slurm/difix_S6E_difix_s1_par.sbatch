#!/bin/bash
#SBATCH --job-name=difix_S6E_difix_s1
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/difix_S6E_difix_s1_par_%j.log
module load cuda/12.6 2>/dev/null || true
V=/work/doreen071/vvc; PY=/home/doreen071/.conda/envs/vvc/bin/python
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $V/work/difix_submission_h200.py $V/submissions/S6E_ens/renders $V/submissions/S6E_difix_s1/renders --shift 320,288 > $V/jobs/difix_S6E_difix_s1_par_shard$i.log 2>&1 &
done; wait
echo "S6E_difix_s1: $(find $V/submissions/S6E_difix_s1/renders -name '*.jpg' | wc -l)/2056"; echo DIFIX_S6E_difix_s1_PAR_DONE $(date +%H:%M)
