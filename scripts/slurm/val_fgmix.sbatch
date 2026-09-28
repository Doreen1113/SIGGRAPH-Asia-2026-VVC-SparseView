#!/bin/bash
#SBATCH --job-name=val_fgmix
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_fgmix_%j.log
# The person-region composite delivered its FG-SSIM gain on the real test set (+0.0059) but wrecked FG-LPIPS
# (+0.0156), which we attribute to mask-boundary inconsistency. Try the same FG specialist mixed into the
# WHOLE-IMAGE ensemble instead - no mask, so no boundary - and see whether the SSIM gain survives without the
# LPIPS penalty.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY - <<'PYEOF'
import json, numpy as np, cv2
V='/work/doreen071/vvc'; D=f'{V}/fixtest/001_1'; H=f'{V}/fixtest_h200/001_1_seq0'
man=json.load(open(f'{D}/manifest.json'))
for name,w in [('fgm1',1),('fgm2',2)]:
    n=0
    for e in man:
        t=e['tag']
        a=cv2.imread(f'{D}/{t}_render.png').astype(np.float64)*4
        f=cv2.imread(f'{H}/m_FGW40_L60K/{t}_render.png')
        if f is None: print('missing FGW40_L60K val render'); break
        cv2.imwrite(f'{D}/{t}_{name}.png', np.clip((a+f.astype(np.float64)*w)/(4+w),0,255).round().astype(np.uint8)); n+=1
    print('built',name,n)
PYEOF
for MIX in fgm1 fgm2; do
  S=$D/wm_$MIX; mkdir -p $S; cp $D/manifest.json $S/
  for f in $D/*_gt.png $D/*_ref.png; do ln -sf $f $S/; done
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
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $S repo199tta pAVt_l15 --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $i/8 > /dev/null 2>&1 & done; wait
  i=0
  for PA in 0.60 0.70 0.80; do
    TAG=$(printf "m%03d" $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S repo199tta $PA pAVt_l15 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1))
  done; wait
  echo "=== 整張圖混入 FGW40_L60K x${MIX#fgm} (無遮罩) ==="
  $PY $W/metrics_both.py $S m060 m070 m080 2>&1 | grep -vE "Warning|warn|load_state"
done
echo "--- 對照：目前送出的版本 sub070 = 26.345/0.9267/0.1495 | 24.023/0.8616/0.2090 ---"
echo VAL_FGMIX_DONE
