#!/bin/bash
#SBATCH --job-name=fit_fgprior
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --exclude=25a-hgpn144,25a-hgpn111
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/work/doreen071/vvc/jobs/fit_fgprior_%j.log
V=/work/doreen071/vvc; PY=/home/doreen071/.conda/envs/vvc/bin/python
$PY $V/work/camcolor_fg.py fit $V/fixtest/012_0 '{D}/ps_lev/{tag}_f101.png' $V/work/priors/fg_012_0.json
$PY $V/work/camcolor_fg.py fit $V/fixtest/001_1 '{D}/ps_cur/{tag}_c_x_tp15.png' $V/work/priors/fg_001_1.json
$PY - <<'PYEOF'
import json, numpy as np
a=json.load(open('/work/doreen071/vvc/work/priors/fg_012_0.json'))
b=json.load(open('/work/doreen071/vvc/work/priors/fg_001_1.json'))
m={v: np.mean([a[v], b[v]], 0).tolist() for v in a if v in b}
json.dump(m, open('/work/doreen071/vvc/work/priors/fg_merged.json','w'), indent=1)
print('merged', {v: np.round(p,2).tolist() for v,p in m.items()})
PYEOF
echo FIT_FGPRIOR_DONE
