#!/bin/bash
#SBATCH --job-name=val_jpegq2
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/work/doreen071/vvc/jobs/val_jpegq2_%j.log
# Clean re-test of JPEG quality as a post-process. Source = lossless composite PNG (sub070). Each variant is ONE
# encode of that source. Sanity: PSNR(variant, source) must rise monotonically with q; if it does not, the harness
# is broken and the GT metrics below are meaningless.
V=/work/doreen071/vvc; D=$V/fixtest/001_1; PY=/home/doreen071/.conda/envs/vvc/bin/python
$PY - <<'PYEOF'
import json, cv2, numpy as np
D='/work/doreen071/vvc/fixtest/001_1'
man=json.load(open(D+'/manifest.json'))
V444=0x111111
variants={'k85':(85,None),'k90':(90,None),'k95':(95,None),'k98':(98,None),'k100':(100,None),'k95f':(95,V444),'k100f':(100,V444)}
acc={k:[] for k in variants}; sz={k:[] for k in variants}
for e in man:
    t=e['tag']; src=cv2.imread(f'{D}/{t}_sub070.png')
    for k,(q,samp) in variants.items():
        params=[cv2.IMWRITE_JPEG_QUALITY,q]+([cv2.IMWRITE_JPEG_SAMPLING_FACTOR,samp] if samp else [])
        ok,buf=cv2.imencode('.jpg',src,params); dec=cv2.imdecode(buf,1)
        cv2.imwrite(f'{D}/{t}_{k}.png',dec)
        mse=np.mean((dec.astype(np.float64)-src.astype(np.float64))**2); acc[k].append(10*np.log10(255**2/mse)); sz[k].append(len(buf)/1e6)
print("sanity PSNR(variant vs lossless source)  mean MB/img")
for k in variants: print(f"  {k:<6} {np.mean(acc[k]):6.2f} dB   {np.mean(sz[k]):.2f} MB")
PYEOF
echo "variant                     FULL psnr/ssim/alex/vgg                FG psnr/ssim/alex/vgg"
$PY $V/work/metrics_both.py $D sub070 k85 k90 k95 k98 k100 k95f k100f 2>&1 | grep -vE "Warning|warn|load_state"
echo VAL_JPEGQ2_DONE
