#!/bin/bash
#SBATCH --job-name=c24_metaens
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/work/doreen071/vvc/jobs/c24_metaens_%j.log
# Pixel-average of already-scored final packages (different model mixes -> different Difix TTA runs on the
# same views). Pure post-processing, no retraining: an extra layer of noise averaging across independent
# pipelines. C13 is the current best and gets the most weight; C12/S1_splice add DIFFERENT model/composite
# choices per scene, so their errors are less correlated with C13's than another copy of the same pipeline.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
$PY $W/meta_ensemble.py $S/C24_meta.zip $S/C13.zip:3 $S/C12_patcam.zip:1 $S/S1_splice.zip:1
chmod 444 $S/C24_meta.zip
echo C24_META_DONE $(date +%H:%M)
