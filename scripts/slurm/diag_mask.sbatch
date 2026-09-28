#!/bin/bash
#SBATCH --job-name=diag_mask
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/diag_mask_%j.log
/home/doreen071/.conda/envs/vvc/bin/python /work/doreen071/vvc/work/maskdiag.py /work/doreen071/vvc/jobs/diag 2>&1 | grep -vE "Warning|warn"
echo DIAG_DONE
