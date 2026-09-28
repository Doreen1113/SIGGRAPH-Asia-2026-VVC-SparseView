#!/bin/bash
#SBATCH --job-name=difix_FGE_difix_s1
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/difix_FGE_difix_s1_%j.log
module load cuda/12.6 2>/dev/null || true
V=/work/doreen071/vvc; PY=/home/doreen071/.conda/envs/vvc/bin/python
SRC=$V/submissions/FGE_ens/renders; DST=$V/submissions/FGE_difix_s1/renders; mkdir -p $DST
until grep -q FGE_ENS_DONE $V/jobs/fge_ens_*.log 2>/dev/null; do sleep 30; done
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i DIFIX_SHARD=$i/8 $PY $V/work/difix_submission_h200.py $SRC $DST --shift 320,288 > $V/jobs/difix_FGE_difix_s1_shard$i.log 2>&1 &
done; wait
echo "FGE_difix_s1: $(find $DST -name '*.jpg' | wc -l)/2056"; echo DIFIX_FGE_difix_s1_DONE $(date +%H:%M)
