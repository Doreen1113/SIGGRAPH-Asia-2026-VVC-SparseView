#!/bin/bash
#SBATCH --job-name=eval_dfx
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/eval_dfx_%j.log
# Is the Difix-distilled model good solo, and does it help the ensemble at weight 1/3/7?
# (FGW20 showed weight 3 beat weight 1 - a strong member can deserve more than an equal share.)
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
V=/work/doreen071/vvc; S=001_1_seq0; R=$V/runs/siga_$S; B=$V/fixtest_h200/$S
PY=/home/doreen071/.conda/envs/vvc/bin/python
echo "== waiting for DFX_s0 on val, $(date -u +%H:%M) =="
until [ -f $R/run_DFX_s0/gaussians.pt ]; do sleep 120; done
[ -f $B/m_DFX_s0/manifest.json ] || python $V/work/dump_val_renders_h200.py $S $R/run_DFX_s0 "" $B/m_DFX_s0 --every 200 > $V/jobs/valm_DFX.log 2>&1
$PY - <<'PYEOF'
import json, numpy as np, cv2
B='/work/doreen071/vvc/fixtest_h200/001_1_seq0'
man=json.load(open(f'{B}/e7/manifest.json'))
for name,w in [('e7_dfx1',1),('e7_dfx3',3),('e7_dfx7',7)]:
    ok=True
    for e in man:
        t=e['tag']
        a=cv2.imread(f'{B}/e7/{t}_render.png').astype(np.float64)*7
        d=cv2.imread(f'{B}/m_DFX_s0/{t}_render.png')
        if d is None: ok=False; break
        cv2.imwrite(f'{B}/e7/{t}_{name}.png', np.clip((a+d.astype(np.float64)*w)/(7+w),0,255).round().astype(np.uint8))
    print('built', name, ok)
PYEOF
echo "=== DFX solo（對照：FULLRES_s0 基準 24.282/0.9135/0.2065 | 22.402/0.8274/0.2915）==="
$PY $V/work/metrics_both.py $B/m_DFX_s0 render 2>&1 | grep -vE "Warning|warn|load_state"
echo "=== DFX 加入 7 成員 ensemble（對照 render = 純 7 成員）==="
$PY $V/work/metrics_both.py $B/e7 render e7_dfx1 e7_dfx3 e7_dfx7 2>&1 | grep -vE "Warning|warn|load_state"
echo EVAL_DFX_DONE
