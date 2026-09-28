#!/bin/bash
# Usage: work/slurm/difix_shift_template.sh <shift> <outdir_name>
# Generates and submits an 8-GPU sbatch job that runs difix_submission_h200.py sharded 8-way on one H200 node.
set -e
SHIFT="$1"; OUT="$2"
SB=/work/doreen071/vvc/work/slurm/difix_${OUT}.sbatch
cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --job-name=difix_${OUT}
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=03:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/difix_${OUT}_%j.log
module load cuda/12.6 2>/dev/null || true
PY=/home/doreen071/.conda/envs/vvc/bin/python
SRC=/work/doreen071/vvc/submissions/SIX_ens/renders
DST=/work/doreen071/vvc/submissions/${OUT}/renders
mkdir -p \$DST
hostname; nvidia-smi --query-gpu=index,name --format=csv,noheader
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=\$i DIFIX_SHARD=\$i/8 \$PY /work/doreen071/vvc/work/difix_submission_h200.py \$SRC \$DST ${SHIFT:+--shift $SHIFT} \
    > /work/doreen071/vvc/jobs/difix_${OUT}_shard\${i}.log 2>&1 &
done
wait
echo "== integrity check =="
for c in \$(ls \$SRC); do echo "\$c: \$(find \$DST/\$c -name '*.jpg' | wc -l) / \$(find \$SRC/\$c -name '*.jpg' | wc -l)"; done
echo DIFIX_${OUT}_DONE
EOF
sbatch "$SB"
