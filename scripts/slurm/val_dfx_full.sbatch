#!/bin/bash
#SBATCH --job-name=val_dfx_full
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_dfx_full_%j.log
# Exact val analogue of the production DFXP package: base = peer + DFX (2:1), background = projection OF THAT BASE
# (previous val test wrongly used the peer-only projection as background, which under-states the gain),
# person = Difix-TTA of a person source that adds the FG specialists on top.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$D/prod_dfx; mkdir -p $S; cp $D/manifest.json $S/
for f in $D/*_gt.png $D/*_ref.png; do ln -sf $f $S/; done
for f in $D/*_pd2.png; do t=$(basename $f _pd2.png); ln -sf $f $S/${t}_render.png; done
echo "== difix x3 on the DFX base, $(date -u +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $S $vn ${s:+--shift $s} --shard $i/8 > /dev/null 2>&1 & done; wait
done
$PY - <<'PYEOF'
import json, numpy as np, cv2
d='/work/doreen071/vvc/fixtest/001_1/prod_dfx'
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']; ims=[cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2')]
    cv2.imwrite(f'{d}/{t}_repo199tta.png', np.clip(np.mean(ims,0),0,255).round().astype(np.uint8))
PYEOF
echo "== projection lam20 of the DFX base, $(date -u +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $S repo199tta pAVt_l20 --nets alex:1,vgg:0.5 --lam 20 --steps 200 --batch 2 --shard $i/8 > /dev/null 2>&1 & done; wait
echo "== person source: DFX base + FG specialists, $(date -u +%H:%M) =="
$PY - <<'PYEOF'
import json, numpy as np, cv2
V='/work/doreen071/vvc'; D=f'{V}/fixtest/001_1'; S=f'{D}/prod_dfx'; H=f'{V}/fixtest_h200/001_1_seq0'
man=json.load(open(f'{S}/manifest.json'))
MIX={'ps_none':[], 'ps_f40l':[('FGW40_L60K',4)], 'ps_f8l6':[('FGW80_s0',2),('LONG60K_s0',2)], 'ps_f20l':[('FGW20_L60K',4)]}
for name,extra in MIX.items():
    n=0
    for e in man:
        t=e['tag']
        acc=cv2.imread(f'{S}/{t}_render.png').astype(np.float64)*6; tot=6; ok=True
        for src,w in extra:
            im=cv2.imread(f'{H}/m_{src}/{t}_render.png')
            if im is None: ok=False; break
            acc+=im.astype(np.float64)*w; tot+=w
        if not ok: continue
        cv2.imwrite(f'{S}/{t}_{name}.png', np.clip(acc/tot,0,255).round().astype(np.uint8)); n+=1
    print('built', name, n)
PYEOF
for MIX in ps_f40l ps_f8l6 ps_f20l; do
  P=$D/pp_$MIX; mkdir -p $P; cp $S/manifest.json $P/
  for f in $S/*_gt.png $S/*_ref.png $S/*_pAVt_l20.png; do ln -sf $f $P/; done
  for f in $S/*_$MIX.png; do t=$(basename $f _$MIX.png); ln -sf $f $P/${t}_render.png; done
  for s in "" "320,288" "640,192"; do
    case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
    for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $P $vn ${s:+--shift $s} --shard $i/8 > /dev/null 2>&1 & done; wait
  done
  $PY - <<PYEOF
import json, numpy as np, cv2
d="$P"
for e in json.load(open(d+'/manifest.json')):
    t=e['tag']; ims=[cv2.imread(f'{d}/{t}_{v}.png').astype(np.float32) for v in ('repo199','repo199s1','repo199s2')]
    cv2.imwrite(f'{d}/{t}_repo199tta.png', np.clip(np.mean(ims,0),0,255).round().astype(np.uint8))
PYEOF
  i=0
  for PA in 0.40 0.50 0.65; do
    TAG=$(printf "q%03d" $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $P repo199tta $PA pAVt_l20 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1))
  done; wait
  echo "=== person=$MIX (base=DFX, bg=projection of DFX base) ==="
  $PY $W/metrics_both.py $P q040 q050 q065 2>&1 | grep -vE "Warning|warn|load_state"
done
echo "=== no person swap (base DFX only) ==="
i=0
for PA in 0.40 0.50 0.65; do
  TAG=$(printf "q%03d" $(echo "$PA*100/1" | bc))
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S repo199tta $PA pAVt_l20 1.0 $TAG > /dev/null 2>&1 &
  i=$((i+1))
done; wait
$PY $W/metrics_both.py $S q040 q050 q065 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_DFX_FULL_DONE
