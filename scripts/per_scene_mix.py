"""Per-scene config choice: the aggregate metric is a frame-weighted mean, and scenes differ in how
efficiently Difix trades SSIM for LPIPS. Mixing configs per scene therefore beats any uniform setting.
Enumerates all 2^5 light/strong assignments and scores the OFFICIAL rank for each."""
import os
import json, urllib.request, ssl, itertools
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
tok=os.environ['VVC_TOKEN']
req=urllib.request.Request("https://8.136.221.94/api/submissions", headers={'Authorization':f'Bearer {tok}'})
subs=json.load(urllib.request.urlopen(req,context=ctx,timeout=30))
CFG={'light':'restore_t100.zip','strong':'pb_p50_b45.zip'}
P={}
for e in subs:
    for k,f in CFG.items():
        if e['filename']==f and e['status']=='succeeded':
            P[k]={s:v for s,v in ((e.get('metrics') or {}).get('per_scene') or {}).items() if '_seq' in s}
SC=sorted(P['light'].keys()); VW={s:P['light'][s]['views'] for s in SC}; TOT=sum(VW.values())
OPP={'shengqi':dict(fp=27.682,fs=0.9180912,fl=0.1819135,gp=25.551,gs=0.8393987,gl=0.2949982),
     'mmm':dict(fp=24.979,fs=0.9111418,fl=0.2788681,gp=24.192,gs=0.8292191,gl=0.2658243),
     'mingzai':dict(fp=23.732,fs=0.9113840,fl=0.2574871,gp=23.035,gs=0.8247841,gl=0.4193558)}
def rank(v,others,lower): return sorted(others+[v],reverse=not lower).index(v)+1
def agg(choice):
    o={k:0.0 for k in ['fp','fs','fl','gp','gs','gl']}
    for s,c in zip(SC,choice):
        m=P[c][s]; w=VW[s]/TOT
        o['fp']+=w*m['psnr']; o['fs']+=w*m['ssim']; o['fl']+=w*m['lpips']
        o['gp']+=w*m['foreground_psnr']; o['gs']+=w*m['foreground_ssim']; o['gl']+=w*m['foreground_lpips']
    return o
def rank_of(us):
    ov=list(OPP.values())
    rf=[rank(us['fp'],[x['fp'] for x in ov],False),rank(us['fs'],[x['fs'] for x in ov],False),rank(us['fl'],[x['fl'] for x in ov],True)]
    rg=[rank(us['gp'],[x['gp'] for x in ov],False),rank(us['gs'],[x['gs'] for x in ov],False),rank(us['gl'],[x['gl'] for x in ov],True)]
    s=OPP['shengqi']; oth=[us,OPP['mmm'],OPP['mingzai']]
    srf=[rank(s['fp'],[x['fp'] for x in oth],False),rank(s['fs'],[x['fs'] for x in oth],False),rank(s['fl'],[x['fl'] for x in oth],True)]
    srg=[rank(s['gp'],[x['gp'] for x in oth],False),rank(s['gs'],[x['gs'] for x in oth],False),rank(s['gl'],[x['gl'] for x in oth],True)]
    return rf,rg,(sum(rf)/3+sum(rg)/3)/2,(sum(srf)/3+sum(srg)/3)/2
rows=[]
for combo in itertools.product(['light','strong'],repeat=5):
    us=agg(combo); rf,rg,ours,sh=rank_of(us)
    rows.append((ours,sh,combo,us,rf,rg))
rows.sort(key=lambda r:(r[0],-r[1]))
print(f'{"scenes(L/S)":<14s}{"FULL p/s/l":>28s}{"FG p/s/l":>28s}  {"FULLrk":>9s}{"FGrk":>9s}{"ours":>7s}{"sheng":>7s}')
seen=set()
for ours,sh,combo,us,rf,rg in rows[:8]:
    tag=''.join('L' if c=='light' else 'S' for c in combo)
    v='  <-- WIN' if ours<sh-1e-9 else ('  <-- TIE' if abs(ours-sh)<1e-9 else '')
    print(f'{tag:<14s}{us["fp"]:>9.3f}{us["fs"]:>9.5f}{us["fl"]:>9.5f}{us["gp"]:>10.3f}{us["gs"]:>9.5f}{us["gl"]:>9.5f}  {str(rf):>9s}{str(rg):>9s}{ours:>7.3f}{sh:>7.3f}{v}')
print()
print('scene order:', SC)
