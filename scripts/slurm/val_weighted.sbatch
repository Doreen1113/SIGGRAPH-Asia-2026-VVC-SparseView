#!/bin/bash
#SBATCH --job-name=val_weighted
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_weighted_%j.log
# Validate the weighted-ensemble trick on val 001_1: the peer's 4-member average render (fixtest) folded in with
# weight 4 alongside individually rendered H200 members, vs. those members' own plain ensembles.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10
export TORCH_EXTENSIONS_DIR=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121
mkdir -p $TORCH_EXTENSIONS_DIR && cp -r /work/doreen071/vvc/envcache/gsplat_prebuilt/gsplat_cuda $TORCH_EXTENSIONS_DIR/
V=/work/doreen071/vvc; S=001_1_seq0; R=$V/runs/siga_$S
# render each H200 val member alone at the val test views
i=0
for M in G3M_FR_s0 GATE_FR_s0 FULLRES_s1; do
  D=$V/fixtest_h200/$S/m_$M
  [ -f $D/manifest.json ] || CUDA_VISIBLE_DEVICES=$((i%4)) python $V/work/dump_val_renders_h200.py $S $R/run_$M "" $D --every 200 > $V/jobs/valm_$M.log 2>&1 &
  i=$((i+1))
done
wait
PY=/home/doreen071/.conda/envs/vvc/bin/python
# build weighted combinations against the peer's 4-member render already in fixtest/001_1
$PY - <<'PYEOF'
import json, numpy as np, cv2, os
V='/work/doreen071/vvc'; S='001_1_seq0'
old=f'{V}/fixtest/001_1'
man=json.load(open(f'{old}/manifest.json'))
mem=['G3M_FR_s0','GATE_FR_s0','FULLRES_s1']
combos={'w4_plus1':(['G3M_FR_s0'],4), 'w4_plus2':(['G3M_FR_s0','GATE_FR_s0'],4), 'w4_plus3':(mem,4)}
for name,(ms,w) in combos.items():
    for e in man:
        t=e['tag']
        acc=cv2.imread(f'{old}/{t}_render.png').astype(np.float32)*w; tot=w
        ok=True
        for m in ms:
            p=f'{V}/fixtest_h200/{S}/m_{m}/{t}_render.png'
            im=cv2.imread(p)
            if im is None: ok=False; break
            acc+=im.astype(np.float32); tot+=1
        if ok: cv2.imwrite(f'{old}/{t}_{name}.png', np.clip(acc/tot,0,255).round().astype(np.uint8))
    print('built', name)
PYEOF
$PY $V/work/metrics_both.py $old render w4_plus1 w4_plus2 w4_plus3 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_WEIGHTED_DONE
