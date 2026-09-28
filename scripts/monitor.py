import re, time, json, os, glob, sys
from pathlib import Path
V='/home/intern_2603055/vvc'
logs={f'{V}/recipe_P_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P2_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P3_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P4_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P5_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P6_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P7_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P8_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P9_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P10_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P11_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P12_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P9_012_0.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/recipe_P13_001_1.log': r'^==|MEAN|Traceback|Error|!!',
      f'{V}/init_probe.log': r'^==|MEAN|Traceback|Error|!!|DONE',
      f'{V}/runs_012_0_J_seed1.log': r'FTGSPP pipeline completed|Traceback|Error',
      f'{V}/p8_chain.log': r'START|DONE|Traceback|Error|!!',
      f'{V}/recipe_012_0_pseudo.log': r'^==|MEAN|Traceback|Error|!!|fail',
      f'{V}/points_selfcap.log': r'DONE|Traceback|Error',
      f'{V}/recipe_selfcap.log': r'^==|DONE|Traceback|Error|!!',
      f'{V}/upload_F_ens3.log': r'^==|HTTP|reserve|Error|failed',
      f'{V}/final_F_ens3.log': r'FINAL|!!|Traceback',
      f'{V}/work/seva/gen_test_cases.log': r'^==|fail|Traceback|Error|scenes built',
      f'{V}/recipe_F_s1.log': r'RECIPE|!!', f'{V}/recipe_F_s2.log': r'RECIPE|!!', f'{V}/recipe_F_s0.log': r'RECIPE|!!'}
seen={k:0 for k in logs}
evals=[f'{V}/runs/siga_001_1_seq0/run_H_F60k/eval_full_s0.5.json', f'{V}/runs/siga_001_1_seq0/run_I_F_scale1/eval_full_s0.5.json', f'{V}/runs/siga_001_1_seq0/run_P/eval_full_s0.5.json', f'{V}/runs/siga_001_1_seq0/run_P2/eval_full_s0.5.json', f'{V}/runs/siga_001_1_seq0/run_P3/eval_full_s0.5.json', f'{V}/runs/siga_001_1_seq0/run_P4/eval_full_s0.5.json', f'{V}/runs/siga_001_1_seq0/run_P5/eval_full_s0.5.json', f'{V}/runs/siga_001_1_seq0/run_P6/eval_full_s0.5.json', f'{V}/runs/siga_001_1_seq0/run_P7/eval_full_s0.5.json', f'{V}/runs/siga_001_1_seq0/run_P8/eval_full_s0.5.json', f'{V}/runs/siga_001_1_seq0/run_P9/eval_full_s0.5.json', f'{V}/runs/siga_012_0_seq0/run_P8/eval_full_s0.5.json']
seen_eval=set(); tb_seen=set()
def emit(s): print(s, flush=True)
while True:
    for f,pat in logs.items():
        try: lines=open(f,errors='ignore').read().splitlines()
        except Exception: continue
        for l in lines[seen[f]:]:
            if re.search(pat,l): emit(f'[{Path(f).name}] {l[:200]}')
        seen[f]=len(lines)
    for e in evals:
        if e not in seen_eval and os.path.exists(e):
            try: d=json.load(open(e)); emit(f'[EVAL] {e.split("/")[-2]}: psnr={d["mean"][0]:.3f} ssim={d["mean"][1]:.4f} lpips={d["mean"][2]:.4f}'); seen_eval.add(e)
            except Exception: pass
    for f in glob.glob(f'{V}/runs_*.log')+[f'{V}/work/seva/run_001_1_batch.log']:
        if f in tb_seen: continue
        try:
            t=open(f,'rb').read()[-4000:].decode(errors='ignore')
            if 'Traceback' in t or 'OutOfMemoryError' in t: emit(f'[CRASH] {Path(f).name}: '+t.strip().splitlines()[-1][:200]); tb_seen.add(f)
        except Exception: pass
    time.sleep(60)
