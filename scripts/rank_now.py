"""Recompute the official rank = (Rank_Full + Rank_FG)/2 from the LIVE leaderboard for every team."""
import json, urllib.request, ssl
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
d=json.load(urllib.request.urlopen("https://8.136.221.94/api/leaderboard?track=sparse_views",context=ctx,timeout=30))
T={}
for e in d:
    m=e['metrics']
    if m.get('foreground_psnr') is None: continue
    T[e['display_name']]=dict(fp=m['psnr'],fs=m['ssim'],fl=m['lpips'],
                              gp=m['foreground_psnr'],gs=m['foreground_ssim'],gl=m['foreground_lpips'])
def rk(name,key,lower):
    vals=sorted((v[key] for v in T.values()), reverse=not lower)
    return vals.index(T[name][key])+1
print(f'  {"team":<14s}{"FULL ranks":>14s}{"FG ranks":>12s}{"FINAL":>8s}   (teams with FG data: %d)'%len(T))
out=[]
for n in T:
    rf=[rk(n,'fp',False),rk(n,'fs',False),rk(n,'fl',True)]
    rg=[rk(n,'gp',False),rk(n,'gs',False),rk(n,'gl',True)]
    out.append(((sum(rf)/3+sum(rg)/3)/2, n, rf, rg))
for fin,n,rf,rg in sorted(out):
    print(f'  {n:<14s}{str(rf):>14s}{str(rg):>12s}{fin:>8.3f}')
