#!/bin/bash
#SBATCH --job-name=val_pswap
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_pswap_%j.log
# Person-region model swap on val 001_1: background stays the peer's render (protects FULL), the PERSON region
# comes from a mix of the peer's render with FG-specialised H200 models. Sidesteps the whole-image dilution
# measured on the real test set (NINE_p50_b45).
V=/work/doreen071/vvc; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
$PY - <<'PYEOF'
import json, numpy as np, cv2
V='/work/doreen071/vvc'; D=f'{V}/fixtest/001_1'; H=f'{V}/fixtest_h200/001_1_seq0'
man=json.load(open(f'{D}/manifest.json'))
def rd(p):
    im=cv2.imread(p); return None if im is None else im.astype(np.float64)
# person-source mixes: peer 4-member render has weight 4 (it is a mean of 4)
MIX={'ps_f1':[('peer',4),('FGW20_s0',1)], 'ps_f2':[('peer',4),('FGW20_s0',2)], 'ps_f4':[('peer',4),('FGW20_s0',4)],
     'ps_l1':[('peer',4),('LONG60K_s0',1)], 'ps_l2':[('peer',4),('LONG60K_s0',2)],
     'ps_fl1':[('peer',4),('FGW20_s0',1),('LONG60K_s0',1)], 'ps_fl2':[('peer',4),('FGW20_s0',2),('LONG60K_s0',2)],
     'ps_fl4':[('peer',4),('FGW20_s0',4),('LONG60K_s0',4)], 'ps_only':[('FGW20_s0',1),('LONG60K_s0',1)]}
for name,parts in MIX.items():
    n=0
    for e in man:
        t=e['tag']; acc=None; tot=0; ok=True
        for src,w in parts:
            im=rd(f'{D}/{t}_render.png' if src=='peer' else f'{H}/m_{src}/{t}_render.png')
            if im is None: ok=False; break
            acc=im*w if acc is None else acc+im*w; tot+=w
        if not ok: continue
        cv2.imwrite(f'{D}/{t}_{name}.png', np.clip(acc/tot,0,255).round().astype(np.uint8)); n+=1
    print('built', name, n)
PYEOF
# composite: person region (soft DeepLab mask on the raw peer render) <- mix, background <- raw peer render
i=0
for M in ps_f1 ps_f2 ps_f4 ps_l1 ps_l2 ps_fl1 ps_fl2 ps_fl4 ps_only; do
  CUDA_VISIBLE_DEVICES=$((i%2)) $PY $V/work/composite2.py dump $D $M 1.0 render 0.0 sw_$M > /dev/null 2>&1 &
  i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
done
wait
echo "=== person-region swap on val 001_1 (render = peer 4-member, untouched) ==="
$PY $V/work/metrics_both.py $D render sw_ps_f1 sw_ps_f2 sw_ps_f4 sw_ps_l1 sw_ps_l2 sw_ps_fl1 sw_ps_fl2 sw_ps_fl4 sw_ps_only 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_PSWAP_DONE
