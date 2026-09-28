#!/bin/bash
#SBATCH --job-name=pseudo_nb
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/pseudo_nb_%j.log
V=/work/doreen071/vvc; PY=/home/doreen071/.conda/envs/vvc312/bin/python
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do
  rm -rf $V/pseudo_nb/$c
  $PY $V/work/tta_to_pseudo.py $c $V/submissions/NB_TTA/renders $V/pseudo_nb/$c --every 2 --scale 0.5 > $V/jobs/pseudo_nb_$c.log 2>&1 &
done; wait
for c in 004_1_seq0 006_1_seq0 007_0_seq0 009_0_seq0 011_0_seq0; do echo "$c: $([ -f $V/pseudo_nb/$c/index.json ] && ls $V/pseudo_nb/$c/img | wc -l || echo MISSING)"; done
echo PSEUDO_NB_DONE $(date +%H:%M)
