#!/bin/bash
#SBATCH --job-name=val_fge2
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_fge2_%j.log
# Follow-up to val_fgens: (a) higher person alphas on the 5-member FG ensemble (the LPIPS trend was still
# improving at 0.85), (b) a 9-member FG ensemble to see whether more members keep helping or start diluting.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# (a) extra alphas on the existing fe0 dump (Difix TTA already built there)
S=$D/fe_fe0
i=0
for PA in 0.92 1.00; do
  TAG=$(printf "e%03d" $(echo "$PA*100/1" | bc))
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S repo199tta $PA pAVt_l15 1.0 $TAG > /dev/null 2>&1 &
  i=$((i+1))
done; wait
echo "=== (a) 5 成員 FG ensemble，更高 person alpha ==="
$PY $W/metrics_both.py $S e085 e092 e100 2>&1 | grep -vE "Warning|warn|load_state"
# (b) 9-member FG ensemble
$PY - <<'PYEOF'
import json, numpy as np, cv2
V='/work/doreen071/vvc'; D=f'{V}/fixtest/001_1'; H=f'{V}/fixtest_h200/001_1_seq0'
man=json.load(open(f'{D}/manifest.json'))
FG9=['FGW40_L60K','FGW20_L60K','LONG90K_s0','FGW80_s0','FGW20_s0','FGW40_s0','FGW10_s0','LONG60K_s0','FGW20_D16']
n=0
for e in man:
    t=e['tag']; fs=[]
    for m in FG9:
        im=cv2.imread(f'{H}/m_{m}/{t}_render.png')
        if im is None: print('missing',m); break
        fs.append(im.astype(np.float64))
    if len(fs)<len(FG9): continue
    cv2.imwrite(f'{D}/{t}_fe9.png', np.clip(np.mean(fs,0),0,255).round().astype(np.uint8)); n+=1
print('built fe9',n,'from',len(FG9),'models')
PYEOF
S=$D/fe_fe9; mkdir -p $S; cp $D/manifest.json $S/
for f in $D/*_gt.png $D/*_ref.png $D/*_pAVt_l15.png; do ln -sf $f $S/; done
for f in $D/*_fe9.png; do t=$(basename $f _fe9.png); ln -sf $f $S/${t}_render.png; done
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
for PA in 0.70 0.85 0.92; do
  TAG=$(printf "e%03d" $(echo "$PA*100/1" | bc))
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S repo199tta $PA pAVt_l15 1.0 $TAG > /dev/null 2>&1 &
  i=$((i+1))
done; wait
echo "=== (b) 9 成員 FG ensemble ==="
$PY $W/metrics_both.py $S e070 e085 e092 2>&1 | grep -vE "Warning|warn|load_state"
echo "--- 對照 sub070 = 26.345/0.9267/0.1495 | 24.023/0.8616/0.2090 ; fe0 e085 = 26.478/0.9277/0.1472 | 24.655/0.8708/0.1877 ---"
echo VAL_FGE2_DONE
