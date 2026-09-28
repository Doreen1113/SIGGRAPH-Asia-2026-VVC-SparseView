#!/bin/bash
#SBATCH --job-name=fgssim_rank
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fgssim_rank_%j.log
# Opponents (awesomegs 0.84229, mingjian 0.83610) pushed FG-SSIM past us. Post-processing is exhausted, so find
# which TRAINING config gives the highest solo FG-SSIM and build the next ensemble around that one.
V=/work/doreen071/vvc; S=001_1_seq0; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
echo "model            FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
for D in $V/fixtest_h200/$S/m_*; do
  M=$(basename $D); [ -f $D/manifest.json ] || continue
  printf "%-16s " "${M#m_}"
  $PY $V/work/metrics_both.py $D render 2>&1 | grep -vE "Warning|warn|load_state|variant" | tail -1
done
echo FGSSIM_RANK_DONE
