#!/bin/bash
#SBATCH --job-name=val_colorvar
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_colorvariants_%j.log
/home/doreen071/.conda/envs/vvc/bin/python /work/doreen071/vvc/work/val_colorvariants.py
