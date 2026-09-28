#!/bin/bash
#SBATCH --job-name=val_dump_e10
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_dump_e10_%j.log
# Test-scene submissions are every 10th frame from fixed cameras. Dump the val scene the same way so a temporal
# background average can be evaluated against GT exactly as it would be applied to the test renders.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
V=/work/doreen071/vvc; S=001_1_seq0; R=$V/runs/siga_$S
CUDA_VISIBLE_DEVICES=0 python $V/work/dump_val_renders_h200.py $S $R/run_S5_FGW40_L60K "" $V/fixtest_h200/$S/s5_e10 --every 10 > $V/jobs/valdump_s5_e10.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python $V/work/dump_val_renders_h200.py $S $R/run_S5_FGW40_L60K "$R/run_FGW20_L60K,$R/run_LONG90K_s0,$R/run_FGW80_s0,$R/run_DFX_s0" $V/fixtest_h200/$S/ens5_e10 --every 10 > $V/jobs/valdump_ens5_e10.log 2>&1 &
wait
for d in s5_e10 ens5_e10; do echo "$d: $(ls $V/fixtest_h200/$S/$d/*_render.png 2>/dev/null | wc -l) frames"; done
echo VAL_DUMP_E10_DONE $(date +%H:%M)
