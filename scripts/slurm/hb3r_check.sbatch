#!/bin/bash
#SBATCH --job-name=hb3r_check
#SBATCH --account=GOV114009
#SBATCH --partition=8gpus
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --output=/work/doreen071/vvc/jobs/hb3r_check_%j.log
# Wait for the three HB3 Difix shifts and SIX Difix shifts, build both TTAs, then prove the rebuild is faithful:
# composite rebuilt HB3 at p085 over SIX_pAVt15 (the exact recipe of the scored HB3_p085_six.zip) and diff vs the zip.
V=/work/doreen071/vvc; W=$V/work; S=$V/submissions; PY=/home/doreen071/.conda/envs/vvc/bin/python
export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cnt() { find $1 -name '*.jpg' 2>/dev/null | wc -l; }
until [ "$(cnt $S/HB3_difix)" -eq 2056 ] && [ "$(cnt $S/HB3_difix_s1)" -eq 2056 ] && [ "$(cnt $S/HB3_difix_s2)" -eq 2056 ]; do sleep 60; done
echo "== HB3_TTA, $(date +%H:%M) =="
$PY $W/tta_average.py $S/HB3_TTA/renders $S/HB3_difix/renders $S/HB3_difix_s1/renders $S/HB3_difix_s2/renders --procs 90
echo "HB3_TTA: $(cnt $S/HB3_TTA)/2056"
echo "== faithfulness check (004 view 02 + 011 view 38, 6 frames) =="
$PY - <<'PYEOF'
import sys, zipfile, os, numpy as np, cv2, torch
sys.argv=['x','tree']
V='/work/doreen071/vvc'; S=V+'/submissions'
import importlib.util
spec=importlib.util.spec_from_file_location('c2', V+'/work/composite2.py')
PYEOF
mkdir -p $S/HB3R_chk
for c in 004_1_seq0 011_0_seq0; do
  CUDA_VISIBLE_DEVICES=0 $PY $W/composite2.py tree $S/HB3_ens/renders $S/HB3_TTA/renders 0.85 $S/SIX_pAVt15/renders 1.0 $S/HB3R_chk/renders --cases $c > $V/jobs/hb3r_chk_$c.log 2>&1
done
$PY - <<'PYEOF'
import zipfile, os, numpy as np, cv2
V='/work/doreen071/vvc'; z=zipfile.ZipFile(V+'/submissions/HB3_p085_six.zip'); R=V+'/submissions/HB3R_chk/renders'
for c in ['004_1_seq0','011_0_seq0']:
    diffs=[]
    for v in sorted(os.listdir(f'{R}/{c}'))[:3]:
        for f in sorted(os.listdir(f'{R}/{c}/{v}'))[::15][:3]:
            a=cv2.imread(f'{R}/{c}/{v}/{f}').astype(np.float32)
            b=cv2.imdecode(np.frombuffer(z.read(f'sparseViewTrack/renders/{c}/{v}/{f}'),np.uint8),1).astype(np.float32)
            diffs.append(np.abs(a-b).mean())
    print(c, 'rebuilt vs scored zip: mean |diff| per image =', [round(x,3) for x in diffs])
PYEOF
echo HB3R_CHECK_DONE $(date +%H:%M)
