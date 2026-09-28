#!/bin/bash
#SBATCH --job-name=val_bgv2
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_bgv2_%j.log
# Deployable-only background variants (test-side raw mix members SIX_ens/SIX_TTA were lost in the quota clean, so
# only output-level averages of EXISTING projected trees can ship: NBX15_pAV, NBX10_pAV, SIX_pAVt15).
# Val counterparts: xl15/xl10 = cross-target projection of the mix raw (bg_p4x2d2s) toward the peer TTA;
# pAVt_l15 = peer own-target (exists at top level). Person = S5 Difix TTA (rebuilt here), alpha 0.85.
# Also the never-swept feather width, on the xl15 baseline.
V=/work/doreen071/vvc; W=$V/work; D=$V/fixtest/001_1; H=$V/fixtest_h200/001_1_seq0; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$D/ps_bgv2; mkdir -p $P; cp $D/manifest.json $P/
for f in $D/*_bg_p4x2d2s.png; do t=$(basename $f _bg_p4x2d2s.png); ln -sf $f $P/${t}_render.png; done
for f in $D/*_repo199tta.png; do t=$(basename $f _repo199tta.png); ln -sf $f $P/${t}_ptta.png; done
echo "proj stage: render=$(ls $P/*_render.png|wc -l) ptta=$(ls $P/*_ptta.png|wc -l)"
echo "== projections lam15 / lam10, $(date +%H:%M) =="
for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $P ptta xl15 --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $i/8 > $V/jobs/valbgv2_xl15_$i.log 2>&1 & done; wait
for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $P ptta xl10 --nets alex:1,vgg:0.5 --lam 10 --steps 200 --batch 2 --shard $i/8 > $V/jobs/valbgv2_xl10_$i.log 2>&1 & done; wait
echo "proj done: xl15=$(ls $P/*_xl15.png|wc -l) xl10=$(ls $P/*_xl10.png|wc -l) $(date +%H:%M)"
# person source: S5 render -> Difix 3 shifts -> TTA
C=$D/ps_cur; mkdir -p $C; cp $D/manifest.json $C/
for f in $D/*_gt.png $D/*_ref.png; do ln -sf $f $C/; done
for f in $H/m_S5_FGW40_L60K/*_render.png; do t=$(basename $f _render.png); ln -sf $f $C/${t}_render.png; done
echo "== difix 3 shifts on S5, $(date +%H:%M) =="
for s in "" "320,288" "640,192"; do
  case "$s" in "") vn=repo199;; "320,288") vn=repo199s1;; *) vn=repo199s2;; esac
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $C $vn ${s:+--shift $s} --shard $i/8 > /dev/null 2>&1 & done; wait
done
$PY - "$C" "$P" "$D" <<'PYEOF'
import sys, json, numpy as np, cv2
C, P, D = sys.argv[1:4]
def rd(p): return cv2.imread(p).astype(np.float32)
def wr(p, a): cv2.imwrite(p, np.clip(a, 0, 255).round().astype(np.uint8))
for e in json.load(open(C+'/manifest.json')):
    t = e['tag']
    wr(f'{C}/{t}_tta.png', np.mean([rd(f'{C}/{t}_{v}.png') for v in ('repo199','repo199s1','repo199s2')], 0))
    x15, x10, pl15 = rd(f'{P}/{t}_xl15.png'), rd(f'{P}/{t}_xl10.png'), rd(f'{D}/{t}_pAVt_l15.png')
    wr(f'{C}/{t}_bg_xl15.png', x15); wr(f'{C}/{t}_bg_xl10.png', x10); wr(f'{C}/{t}_bg_pl15.png', pl15)
    wr(f'{C}/{t}_bg_m1510.png', (x15+x10)/2)                 # mean(NBX15, NBX10)      -> deployable
    wr(f'{C}/{t}_bg_m15p.png', (x15+pl15)/2)                 # mean(NBX15, SIX_pAVt15) -> deployable
    wr(f'{C}/{t}_bg_m3.png', (x15+x10+pl15)/3)               # mean of all three       -> deployable
    wr(f'{C}/{t}_bg_m1510w.png', (2*x15+x10)/3)              # 2:1                     -> deployable
print('bg variants built')
PYEOF
echo "== composites, $(date +%H:%M) =="
i=0
for bg in xl15 xl10 pl15 m1510 m15p m3 m1510w; do
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $C tta 0.85 bg_$bg 1.0 c_$bg > $V/jobs/valbgv2_c_$bg.log 2>&1 &
  i=$((i+1))
done
for FW in 9 15 41 61; do
  TAG=f$(printf "%02d" $FW)
  CUDA_VISIBLE_DEVICES=$((i%8)) $PY $W/composite2.py dump $C tta 0.85 bg_xl15 1.0 $TAG --feather $FW > $V/jobs/valbgv2_$TAG.log 2>&1 &
  i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
done
wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(c_xl15 = 現行配方的 val 代理;m* = 可直接部署的投影樹平均;f* = 羽化寬度,基於 xl15)"
$PY $W/metrics_both.py $C c_xl15 c_xl10 c_pl15 c_m1510 c_m1510w c_m15p c_m3 f09 f15 f41 f61 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_BGV2_DONE $(date +%H:%M)
