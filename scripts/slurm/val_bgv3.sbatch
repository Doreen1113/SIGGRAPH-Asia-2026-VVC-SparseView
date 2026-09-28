#!/bin/bash
#SBATCH --job-name=val_bgv3
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_bgv3_%j.log
# Background v3 = own-model mix only (peer SIX models are gone), projected toward the EXISTING NBX15 background
# (val stand-in: xl15). Fully deployable on test: every member has an M_ tree, D = NBX15_pAV, swap_bg into the zip.
# val_bgmix showed the no-peer raw mix (bg_d2s) is only -0.03 dB vs the peer mix and better on LPIPS; we now have
# many more own members (S5 seeds, S3, NBD, FGW family) to average.
module load cuda/12.6 2>/dev/null
export PATH=/home/doreen071/.conda/envs/vvc312/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=/work/doreen071/vvc; W=$V/work; S=001_1_seq0; R=$V/runs/siga_$S; H=$V/fixtest_h200/$S; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
C=$D/ps_cur; P=$D/ps_bgv3; mkdir -p $P; cp $D/manifest.json $P/
echo "== dump missing val members, $(date +%H:%M) =="
g=0
for m in FGW20_s0 FGW40_L60K FGW40_L60K_s1 FGW80_s0 LONG60K_s0 S3_FGW40_L60K; do
  [ "$(ls $H/m_$m/*_render.png 2>/dev/null | wc -l)" -eq 24 ] && continue
  ( TD=$(mktemp -d /tmp/tex.XXXXXX)/py312_cu121; mkdir -p $TD; cp -r $V/envcache/gsplat_prebuilt/gsplat_cuda $TD/
    TORCH_EXTENSIONS_DIR=$TD CUDA_VISIBLE_DEVICES=$g python $W/dump_val_renders_h200.py $S $R/run_$m "" $H/m_$m --every 200 > $V/jobs/bgv3_dump_$m.log 2>&1; rm -rf $(dirname $TD) ) &
  g=$((g+1))
done; wait
for m in DFX_s0 DFXFG_L60K S5_FGW40_L60K NBD_S5_FG FGW20_s0 FGW40_L60K FGW40_L60K_s1 FGW80_s0 LONG60K_s0 S3_FGW40_L60K; do printf "%s:%s " $m "$(ls $H/m_$m/*_render.png 2>/dev/null | wc -l)"; done; echo
$PY - "$P" "$H" "$D" <<'PYEOF'
import sys, json, numpy as np, cv2
P, H, D = sys.argv[1:4]
man = json.load(open(P + '/manifest.json'))
def rd(m, t): return cv2.imread(f'{H}/m_{m}/{t}_render.png').astype(np.float64)
T10 = ['FGW40_L60K', 'FGW40_L60K_s1', 'FGW20_s0', 'FGW80_s0', 'LONG60K_s0', 'S3_FGW40_L60K']
MIX = {'own6':   [('DFX_s0', 2), ('DFXFG_L60K', 2), ('S5_FGW40_L60K', 2), ('NBD_S5_FG', 2), ('T10', 2)],
       'own6nn': [('DFX_s0', 2), ('DFXFG_L60K', 2), ('S5_FGW40_L60K', 2), ('T10', 2)],
       'ownT':   [('DFX_s0', 2), ('DFXFG_L60K', 2), ('S5_FGW40_L60K', 2), ('NBD_S5_FG', 2), ('T10', 4)],
       'ownD':   [('DFX_s0', 3), ('DFXFG_L60K', 2), ('S5_FGW40_L60K', 1), ('NBD_S5_FG', 3), ('T10', 1)]}
for e in man:
    t = e['tag']; t10 = np.mean([rd(m, t) for m in T10], 0)
    for name, parts in MIX.items():
        acc = 0; tot = 0
        for src, w in parts:
            acc = acc + (t10 if src == 'T10' else rd(src, t)) * w; tot += w
        cv2.imwrite(f'{P}/{t}_{name}.png', np.clip(acc / tot, 0, 255).round().astype(np.uint8))
    cv2.imwrite(f'{P}/{t}_old.png', cv2.imread(f'{D}/{t}_bg_p4x2d2s.png'))
print('mixes built')
PYEOF
for f in $C/*_bg_xl15.png; do t=$(basename $f _bg_xl15.png); ln -sf $f $P/${t}_xl15.png; done
echo "== projections toward xl15 (deployable: D = NBX15_pAV on test), $(date +%H:%M) =="
for mix in own6 own6nn ownT ownD old; do
  for f in $P/*_$mix.png; do t=$(basename $f _$mix.png); ln -sf $f $P/${t}_render.png; done   # R = this mix
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/proj_multi.py dump $P xl15 x_$mix --nets alex:1,vgg:0.5 --lam 15 --steps 200 --batch 2 --shard $i/8 > $V/jobs/bgv3_proj_${mix}_$i.log 2>&1 & done; wait
  echo "  $mix -> $(ls $P/*_x_$mix.png | wc -l)/24 $(date +%H:%M)"
done
g=0
for mix in own6 own6nn ownT ownD old; do
  for f in $P/*_x_$mix.png; do t=$(basename $f _x_$mix.png); ln -sf $f $C/${t}_bg_x_$mix.png; done
  CUDA_VISIBLE_DEVICES=$((g%8)) $PY $W/composite2.py dump $C tta 0.85 bg_x_$mix 1.0 c_x_$mix > $V/jobs/bgv3_comp_$mix.log 2>&1 &
  g=$((g+1))
done; wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(c_xl15 = 現行背景;c_x_old = 舊混合重新投影向 xl15(對照);c_x_own* = 純自家成員混合 → 投影向 xl15,可部署)"
$PY $W/metrics_both.py $C c_xl15 c_x_old c_x_own6 c_x_own6nn c_x_ownT c_x_ownD 2>&1 | grep -vE "Warning|warn|load_state"
echo "--- raw mixes (no projection, no person) vs old raw mix, for reference"
$PY $W/metrics_both.py $P old own6 own6nn ownT ownD 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_BGV3_DONE $(date +%H:%M)
