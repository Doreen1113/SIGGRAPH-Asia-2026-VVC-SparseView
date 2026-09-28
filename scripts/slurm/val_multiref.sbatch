#!/bin/bash
#SBATCH --job-name=val_multiref
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_multiref_%j.log
# HAD (CVPR26) test-time part without its weights: condition Difix on K different reference train views and fuse.
# HAD fuses by per-pixel argmin of a learned hallucination score; weight-free analogues here: mean and median.
# Person source = S5 render (ps_cur), background = xl15, alpha 0.85, feather 25 -> compare with c_xl15 (ref1 x 3 shifts).
V=/work/doreen071/vvc; W=$V/work; C=$V/fixtest/001_1/ps_cur; DATA=$V/data/001_1_seq0; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY - "$C" "$DATA" <<'PYEOF'
import sys, json, re, numpy as np, cv2
C, DATA = sys.argv[1:3]
txt = open(f'{DATA}/extri.yml').read()
def mat(name):
    m = re.search(name + r': !!opencv-matrix.*?data: \[([^\]]*)\]', txt, re.S); return np.array([float(x) for x in m.group(1).split(',')])
train = ['12', '15', '28', '29', '47', '51']
def center(c): R = mat(f'Rot_{c}').reshape(3, 3); T = mat(f'T_{c}'); return -R.T @ T
def fwd(c): return mat(f'Rot_{c}').reshape(3, 3)[2]
man = json.load(open(f'{C}/manifest.json')); order = {}
for e in man:
    v = e['view']
    if v not in order:
        cv = center(v); fv = fwd(v)
        order[v] = sorted(train, key=lambda t: np.linalg.norm(center(t) - cv))
        print(v, 'nearest by position:', order[v][:3], 'manifest ref_cam:', e['ref_cam'], ' dir-cos:', [round(float(fwd(t) @ fv), 3) for t in order[v][:3]])
    for k in (1, 2):
        cam = order[v][k]; img = cv2.imread(f"{DATA}/images/{cam}/{e['frame']:06d}.jpg")
        cv2.imwrite(f"{C}/{e['tag']}_ref{k+1}.png", img)
print('refs staged')
PYEOF
echo "== difix ref2/ref3 x 3 shifts, $(date +%H:%M) =="
for R in ref2 ref3; do
  for s in "" "320,288" "640,192"; do
    case "$s" in "") vn=${R}d;; "320,288") vn=${R}d_s1;; *) vn=${R}d_s2;; esac
    for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i $PY $W/difix_dump_h200.py $C $vn --ref $R ${s:+--shift $s} --shard $i/8 > /dev/null 2>&1 & done; wait
  done
  echo "  $R done $(date +%H:%M)"
done
$PY - "$C" <<'PYEOF'
import sys, json, numpy as np, cv2
C = sys.argv[1]
def rd(t, v): return cv2.imread(f'{C}/{t}_{v}.png').astype(np.float32)
def wr(t, v, a): cv2.imwrite(f'{C}/{t}_{v}.png', np.clip(a, 0, 255).round().astype(np.uint8))
r1 = ('repo199', 'repo199s1', 'repo199s2'); r2 = ('ref2d', 'ref2d_s1', 'ref2d_s2'); r3 = ('ref3d', 'ref3d_s1', 'ref3d_s2')
for e in json.load(open(C + '/manifest.json')):
    t = e['tag']
    A1 = [rd(t, v) for v in r1]; A2 = [rd(t, v) for v in r2]; A3 = [rd(t, v) for v in r3]
    wr(t, 'mr_mean3', np.mean([A1[0], A2[0], A3[0]], 0))            # 3 refs, no shift
    wr(t, 'mr_mean9', np.mean(A1 + A2 + A3, 0))                     # 3 refs x 3 shifts
    wr(t, 'mr_med9', np.median(np.stack(A1 + A2 + A3), 0))          # robust (HAD-like outlier rejection)
    wr(t, 'mr_mean6', np.mean(A1 + A2, 0))                          # 2 refs x 3 shifts
    wr(t, 'mr_w', (2 * np.mean(A1, 0) + np.mean(A2, 0) + np.mean(A3, 0)) / 4)   # nearest ref weighted x2
    wr(t, 'mr_r2', np.mean(A2, 0)); wr(t, 'mr_r3', np.mean(A3, 0))
print('fused')
PYEOF
g=0
for v in mr_mean3 mr_mean9 mr_med9 mr_mean6 mr_w mr_r2 mr_r3; do
  CUDA_VISIBLE_DEVICES=$((g%8)) $PY $W/composite2.py dump $C $v 0.85 bg_xl15 1.0 c_$v > $V/jobs/valmr_$v.log 2>&1 &
  g=$((g+1))
done; wait
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
echo "(c_xl15 = 現行:最近參照 x 3 位移;mr_r2/r3 = 只用第 2/3 近參照)"
$PY $W/metrics_both.py $C c_xl15 c_mr_mean3 c_mr_mean6 c_mr_mean9 c_mr_med9 c_mr_w c_mr_r2 c_mr_r3 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_MULTIREF_DONE $(date +%H:%M)
