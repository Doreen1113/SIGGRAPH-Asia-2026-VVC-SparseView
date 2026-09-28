#!/bin/bash
#SBATCH --job-name=val_bgmix
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_bgmix_%j.log
# The background has been the peer's 6-member ensemble since day 1; our DFX-family models beat the peer's
# 4-member val ensemble on FULL-PSNR solo (26.43 vs 26.35). Measure mixed backgrounds, raw (before Difix).
V=/work/doreen071/vvc; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
$PY - <<'PYEOF'
import json, numpy as np, cv2
V='/work/doreen071/vvc'; D=f'{V}/fixtest/001_1'; H=f'{V}/fixtest_h200/001_1_seq0'
man=json.load(open(f'{D}/manifest.json'))
MIX={'bg_p4':[('peer',4)],
     'bg_dfx':[('DFX_s0',1)],
     'bg_p4d1':[('peer',4),('DFX_s0',1)],
     'bg_p4d2':[('peer',4),('DFX_s0',1),('DFXFG_L60K',1)],
     'bg_p4d2s':[('peer',4),('DFX_s0',1),('DFXFG_L60K',1),('S5_FGW40_L60K',1)],
     'bg_d2s':[('DFX_s0',1),('DFXFG_L60K',1),('S5_FGW40_L60K',1)],
     'bg_p4x2d2s':[('peer',4),('DFX_s0',2),('DFXFG_L60K',2),('S5_FGW40_L60K',2)]}
for name,parts in MIX.items():
    n=0
    for e in man:
        t=e['tag']; acc=None; tot=0; ok=True
        for src,w in parts:
            p=f'{D}/{t}_render.png' if src=='peer' else f'{H}/m_{src}/{t}_render.png'
            im=cv2.imread(p)
            if im is None: ok=False; print('missing',p); break
            im=im.astype(np.float64); acc=im*w if acc is None else acc+im*w; tot+=w
        if not ok: continue
        cv2.imwrite(f'{D}/{t}_{name}.png', np.clip(acc/tot,0,255).round().astype(np.uint8)); n+=1
    print('built', name, n)
PYEOF
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
$PY $V/work/metrics_both.py $D bg_p4 bg_dfx bg_p4d1 bg_p4d2 bg_p4d2s bg_d2s bg_p4x2d2s 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_BGMIX_DONE
