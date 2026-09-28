#!/bin/bash
#SBATCH --job-name=eval_probe
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/eval_probe_alpha_%j.log
echo "no GT for test scenes -- this job intentionally does nothing; probe_alpha is a visual/statistical diff only"
