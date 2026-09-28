#!/bin/bash
#SBATCH --job-name=fgssim_hunt
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/fgssim_hunt_%j.log
# FG-SSIM is 0.001 short of the rank-2 threshold. Hunt for that last bit: person mixes that include FGW80
# (highest FG-SSIM of the solo models) and a finer person-Difix alpha grid, keeping the projection background.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
$PY - <<'PYEOF'
import json, numpy as np, cv2
V='/work/doreen071/vvc'; D=f'{V}/fixtest/001_1'; H=f'{V}/fixtest_h200/001_1_seq0'
man=json.load(open(f'{D}/manifest.json'))
MIX={'q_fl1':[('peer',4),('FGW20_s0',1),('LONG60K_s0',1)],
     'q_fl2':[('peer',4),('FGW20_s0',2),('LONG60K_s0',2)],
     'q_f8l2':[('peer',4),('FGW80_s0',2),('LONG60K_s0',2)],
     'q_f8l1':[('peer',4),('FGW80_s0',1),('LONG60K_s0',1)],
     'q_f8fl2':[('peer',4),('FGW80_s0',1),('FGW20_s0',1),('LONG60K_s0',2)]}
for name,parts in MIX.items():
    n=0
    for e in man:
        t=e['tag']; acc=None; tot=0; ok=True
        for src,w in parts:
            p=f'{D}/{t}_render.png' if src=='peer' else f'{H}/m_{src}/{t}_render.png'
            im=cv2.imread(p)
            if im is None: ok=False; break
            im=im.astype(np.float64); acc=im*w if acc is None else acc+im*w; tot+=w
        if not ok: continue
        cv2.imwrite(f'{D}/{t}_{name}.png', np.clip(acc/tot,0,255).round().astype(np.uint8)); n+=1
    print('built', name, n)
PYEOF
for MIX in q_fl1 q_fl2 q_f8l2 q_f8l1 q_f8fl2; do
  S=$D/hunt_$MIX; mkdir -p $S; cp $D/manifest.json $S/
  for f in $D/*_gt.png $D/*_ref.png $D/*_pAVt_l15.png; do ln -sf $f $S/; done
  for f in $D/*_$MIX.png; do t=$(basename $f _$MIX.png); ln -sf $f $S/${t}_render.png; done
  for s in "" "320,288" "640,192"; do
    case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
    for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $S $vn ${s:+--shift $s} --shard $i/8 > /dev/null 2>&1 & done; wait
  done
  $PY - <<PYEOF
import json, numpy as np, cv2
d="$S"
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']; ims=[cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2')]
    cv2.imwrite(f'{d}/{t}_repo199tta.png', np.clip(np.mean(ims,0),0,255).round().astype(np.uint8))
PYEOF
  i=0
  for PA in 0.20 0.35 0.50 0.65; do
    TAG=$(printf "h%03d" $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S repo199tta $PA pAVt_l15 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1))
  done; wait
  echo "=== $MIX ==="
  $PY $W/metrics_both.py $S h020 h035 h050 h065 2>&1 | grep -vE "Warning|warn|load_state"
done
echo FGSSIM_HUNT_DONE
