"""After a submission is scored, recompute the OFFICIAL rank from live data and say whether to keep or revert."""
import os
import json, urllib.request, ssl, sys
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
tok=os.environ['VVC_TOKEN']
req=urllib.request.Request("https://8.136.221.94/api/submissions", headers={'Authorization':f'Bearer {tok}'})
subs=json.load(urllib.request.urlopen(req,context=ctx,timeout=30))
lb=json.load(urllib.request.urlopen("https://8.136.221.94/api/leaderboard?track=sparse_views",context=ctx,timeout=30))
OPP={e['display_name']:e['metrics'] for e in lb if e['display_name']!='Doreen071' and e['metrics'].get('foreground_psnr') is not None}
def rank(val, others, lower): return sorted(others+[val], reverse=not lower).index(val)+1
def score(m):
    o=list(OPP.values())
    rf=[rank(m['psnr'],[x['psnr'] for x in o],False), rank(m['ssim'],[x['ssim'] for x in o],False), rank(m['lpips'],[x['lpips'] for x in o],True)]
    rg=[rank(m['foreground_psnr'],[x['foreground_psnr'] for x in o],False), rank(m['foreground_ssim'],[x['foreground_ssim'] for x in o],False), rank(m['foreground_lpips'],[x['foreground_lpips'] for x in o],True)]
    return rf, rg, (sum(rf)/3+sum(rg)/3)/2
print(f'{"submission":<26s}{"FULL":>12s}{"FG":>12s}{"RANK":>8s}')
best=None
for e in subs[:8]:
    m=e.get('metrics') or {}
    if e['status']!='succeeded' or m.get('foreground_psnr') is None: continue
    rf,rg,fin=score(m)
    print(f'{e["filename"]:<26s}{str(rf):>12s}{str(rg):>12s}{fin:>8.3f}')
    if best is None or fin<best[0]: best=(fin,e['filename'])
# opponents' ranks under the current standings
print()
for n,m in OPP.items():
    o=[x for k,x in OPP.items() if k!=n]
    print(f'  {n}: {sum([rank(m["psnr"],[x["psnr"] for x in o],False)])}...(informational)')
print(f'\nBEST OF OUR SCORED SUBMISSIONS: {best[1]} at rank {best[0]:.3f}')
print('Reminder: the official score uses the LATEST valid submission, so the last upload before the deadline must be this one.')
