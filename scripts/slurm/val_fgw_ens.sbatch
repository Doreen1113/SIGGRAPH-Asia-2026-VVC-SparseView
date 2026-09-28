#!/bin/bash
#SBATCH --job-name=val_fgw_ens
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_fgw_ens_%j.log
# Does FGW20 actually help the ensemble, or is it only strong solo? (RESULTS.md warns better-solo != better-ensemble.)
# Reuses the weighted-ensemble identity: e7_render already is the mean of 7 members, so fold it in with weight 7.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
V=/work/doreen071/vvc; S=001_1_seq0; R=$V/runs/siga_$S; B=$V/fixtest_h200/$S
PY=/home/doreen071/.conda/envs/vvc/bin/python
[ -f $B/m_FGW20_s0/manifest.json ] || python $V/work/dump_val_renders_h200.py $S $R/run_FGW20_s0 "" $B/m_FGW20_s0 --every 200 > $V/jobs/valm_FGW20.log 2>&1
$PY - <<'PYEOF'
import json, numpy as np, cv2
B='/work/doreen071/vvc/fixtest_h200/001_1_seq0'
man=json.load(open(f'{B}/e7/manifest.json'))
# e7 is the equal-weight mean of 7 members -> fold in with weight 7 and add FGW20 at weight w
for name,w in [('e7_fgw1',1),('e7_fgw3',3),('e7_fgw7',7)]:
    for e in man:
        t=e['tag']
        a=cv2.imread(f'{B}/e7/{t}_render.png').astype(np.float64)*7
        f=cv2.imread(f'{B}/m_FGW20_s0/{t}_render.png')
        if f is None: print('missing FGW20 render'); break
        cv2.imwrite(f'{B}/e7/{t}_{name}.png', np.clip((a+f.astype(np.float64)*w)/(7+w),0,255).round().astype(np.uint8))
    print('built', name)
PYEOF
echo "=== raw ensemble comparison (no post-processing) ==="
$PY $V/work/metrics_both.py $B/e7 render e7_fgw1 e7_fgw3 e7_fgw7 2>&1 | grep -vE "Warning|warn|load_state"
echo "=== FGW20 solo for reference ==="
$PY $V/work/metrics_both.py $B/m_FGW20_s0 render 2>&1 | grep -vE "Warning|warn|load_state" | tail -1
echo VAL_FGW_ENS_DONE
