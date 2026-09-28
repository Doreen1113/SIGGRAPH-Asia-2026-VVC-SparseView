#!/bin/bash
#SBATCH --job-name=verify_formula
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/verify_formula_%j.log
# Verify the weighted-ensemble identity on val 001_1 with the dumps we already have:
#   reconstructed = (4*e4_render + G3M_s0 + FULLRES_s1) / 6   vs   e6_render (directly rendered 6-member)
# e4/e6 dumps are lossless PNG, so any discrepancy is formula/protocol error, not JPEG noise.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
V=/work/doreen071/vvc; S=001_1_seq0; R=$V/runs/siga_$S
# render the two members that e6 adds on top of e4, individually
for M in G3M_s0 FULLRES_s1; do
  D=$V/fixtest_h200/$S/m_$M
  [ -f $D/manifest.json ] && { echo "have $M"; continue; }
  python $V/work/dump_val_renders_h200.py $S $R/run_$M "" $D --every 200 > $V/jobs/verifym_$M.log 2>&1
done
/home/doreen071/.conda/envs/vvc/bin/python - <<'PYEOF'
import json, numpy as np, cv2
V='/work/doreen071/vvc'; S='001_1_seq0'
B=f'{V}/fixtest_h200/{S}'
man=json.load(open(f'{B}/e6/manifest.json'))
diffs=[]; maxs=[]
for e in man:
    t=e['tag']
    e4=cv2.imread(f'{B}/e4/{t}_render.png').astype(np.float64)
    g3=cv2.imread(f'{B}/m_G3M_s0/{t}_render.png')
    f1=cv2.imread(f'{B}/m_FULLRES_s1/{t}_render.png')
    e6=cv2.imread(f'{B}/e6/{t}_render.png').astype(np.float64)
    if g3 is None or f1 is None: print('missing member render for', t); continue
    recon=(4*e4 + g3.astype(np.float64) + f1.astype(np.float64))/6
    cv2.imwrite(f'{B}/e6/{t}_recon6.png', np.clip(recon,0,255).round().astype(np.uint8))
    d=np.abs(recon-e6)
    diffs.append(d.mean()); maxs.append(d.max())
print(f'formula check over {len(diffs)} images:')
print(f'  mean |recon - direct| = {np.mean(diffs):.4f} levels (out of 255)')
print(f'  max  |recon - direct| = {np.max(maxs):.1f} levels')
print(f'  -> {"FORMULA CONFIRMED (within rounding)" if np.mean(diffs) < 1.0 else "MISMATCH - investigate protocol"}')
PYEOF
/home/doreen071/.conda/envs/vvc/bin/python $V/work/metrics_both.py $V/fixtest_h200/$S/e6 render recon6 2>&1 | grep -vE "Warning|warn|load_state"
echo VERIFY_FORMULA_DONE
