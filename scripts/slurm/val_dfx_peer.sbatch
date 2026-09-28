#!/bin/bash
#SBATCH --job-name=dfx_peer
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/dfx_peer_%j.log
# The decisive question for DFX: adding H200 members to the PEER's base diluted it (NINE_p50_b45, six metrics worse).
# Is DFX different? Test both whole-image mixing and person-region-only mixing against the peer's val render.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0
PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10
$PY - <<'PYEOF'
import json, numpy as np, cv2
V='/work/doreen071/vvc'; D=f'{V}/fixtest/001_1'; H=f'{V}/fixtest_h200/001_1_seq0'
man=json.load(open(f'{D}/manifest.json'))
# peer render is the mean of 4 members -> fold in with weight 4
for name,w in [('pd1',1),('pd2',2),('pd4',4),('pd8',8)]:
    for e in man:
        t=e['tag']
        a=cv2.imread(f'{D}/{t}_render.png').astype(np.float64)*4
        d=cv2.imread(f'{H}/m_DFX_s0/{t}_render.png')
        if d is None: print('missing DFX render'); break
        cv2.imwrite(f'{D}/{t}_{name}.png', np.clip((a+d.astype(np.float64)*w)/(4+w),0,255).round().astype(np.uint8))
    print('built', name)
PYEOF
echo "=== A) 整張圖混入 DFX（raw render 層，未後製）==="
$PY $W/metrics_both.py $D render pd1 pd2 pd4 pd8 2>&1 | grep -vE "Warning|warn|load_state"
# B) full production chain on the best whole-image mix, and on person-region-only
for MIX in pd2 pd4; do
  S=$D/dfxp_$MIX; mkdir -p $S; cp $D/manifest.json $S/
  for f in $D/*_gt.png $D/*_ref.png $D/*_pAVt_l20.png; do ln -sf $f $S/; done
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
  # background from the peer's lambda-20 projection (unchanged), person from this mix's Difix-TTA
  i=0
  for PA in 0.40 0.50 0.65; do
    TAG=$(printf "d%03d" $(echo "$PA*100/1" | bc))
    CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $S repo199tta $PA pAVt_l20 1.0 $TAG > /dev/null 2>&1 &
    i=$((i+1))
  done; wait
  echo "=== B) $MIX : person from Difix-TTA(mix), background = peer projection lam20 ==="
  $PY $W/metrics_both.py $S d040 d050 d065 2>&1 | grep -vE "Warning|warn|load_state"
done
echo DFX_PEER_DONE
