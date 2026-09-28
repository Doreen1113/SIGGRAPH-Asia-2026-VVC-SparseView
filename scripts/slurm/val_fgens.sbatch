#!/bin/bash
#SBATCH --job-name=val_fgens
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_fgens_%j.log
# A single FG specialist folded into the peer base delivered only +0.0059 FG-SSIM on the test set, while the
# model itself is +0.027 better than baseline - most of the gain is lost to dilution and to mask-boundary noise.
# Test whether an ENSEMBLE of FG specialists (averaging kills the per-model noise that LPIPS punishes) used as
# the person source keeps much more of that gain.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY - <<'PYEOF'
import json, numpy as np, cv2
V='/work/doreen071/vvc'; D=f'{V}/fixtest/001_1'; H=f'{V}/fixtest_h200/001_1_seq0'
man=json.load(open(f'{D}/manifest.json'))
FG=['FGW40_L60K','FGW20_L60K','LONG90K_s0','FGW80_s0','FGW20_s0']
# fe0: the FG ensemble on its own;  fe4/fe8: peer base (weight 4) + FG ensemble at weight 4 / 8
for name,pw,fw in [('fe0',0,1),('fe4',4,4),('fe8',4,8)]:
    n=0
    for e in man:
        t=e['tag']; acc=None; tot=0; ok=True
        if pw:
            acc=cv2.imread(f'{D}/{t}_render.png').astype(np.float64)*pw; tot=pw
        fs=[]
        for m in FG:
            im=cv2.imread(f'{H}/m_{m}/{t}_render.png')
            if im is None: ok=False; break
            fs.append(im.astype(np.float64))
        if not ok: break
        fg=np.mean(fs,0)
        acc = fg*fw if acc is None else acc+fg*fw; tot+=fw
        cv2.imwrite(f'{D}/{t}_{name}.png', np.clip(acc/tot,0,255).round().astype(np.uint8)); n+=1
    print('built',name,n,'(FG ensemble of',len(FG),'models)')
PYEOF
for MIX in fe0 fe4 fe8; do
  S=$D/fe_$MIX; mkdir -p $S; cp $D/manifest.json $S/
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
  for PA in 0.50 0.70 0.85; do
    TAG=$(printf "e%03d" $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S repo199tta $PA pAVt_l15 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1))
  done; wait
  echo "=== 人物來源 = $MIX (fe0=純FG ensemble, fe4/fe8=peer4+FGens4/8) ==="
  $PY $W/metrics_both.py $S e050 e070 e085 2>&1 | grep -vE "Warning|warn|load_state"
done
echo "--- 對照 sub070 (目前送出) = 26.345/0.9267/0.1495 | 24.023/0.8616/0.2090 ---"
echo VAL_FGENS_DONE
