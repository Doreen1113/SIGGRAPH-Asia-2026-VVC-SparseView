"""For each of our six metrics, list the value needed to gain each rank, and what FINAL that single flip yields."""
import json, ssl, urllib.request, sys
K=['psnr','ssim','lpips','foreground_psnr','foreground_ssim','foreground_lpips']
NAME={'psnr':'FULL-PSNR','ssim':'FULL-SSIM','lpips':'FULL-LPIPS','foreground_psnr':'FG-PSNR','foreground_ssim':'FG-SSIM','foreground_lpips':'FG-LPIPS'}
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
rows=json.load(urllib.request.urlopen('https://8.136.221.94/api/leaderboard?track=sparse_views',context=ctx,timeout=30))
rows=rows if isinstance(rows,list) else rows.get('entries',[])
teams={}
for r in rows:
    m=r.get('metrics') or {}
    if all(k in m for k in K): teams[r['username']]={k:float(m[k]) for k in K}
teams.pop('doreen071',None)
US=dict(zip(K,[float(x) for x in sys.argv[1:7]]))
def ranks(v):
    out={}
    for k in K:
        hi='lpips' not in k
        vals=[t[k] for t in teams.values()]
        out[k]=1+sum(1 for x in vals if (x>v[k] if hi else x<v[k]))
    return out
R=ranks(US); full=sum(R[k] for k in K[:3])/3; fg=sum(R[k] for k in K[3:])/3
print(f"現在: FULL{[R[k] for k in K[:3]]}={full:.3f}  FG{[R[k] for k in K[3:]]}={fg:.3f}  FINAL={(full+fg)/2:.3f}\n")
print(f"{'指標':<12}{'我們':>10}{'目前名次':>8}   要升到第N名需要的值(以及單獨達成後的FINAL)")
for k in K:
    hi='lpips' not in k
    others=sorted([(n,t[k]) for n,t in teams.items()], key=lambda x:-x[1] if hi else x[1])
    line=f"{NAME[k]:<12}{US[k]:>10.5f}{R[k]:>8}   "
    parts=[]
    for target in range(R[k]-1,0,-1):
        n,v=others[target-1]
        v2=dict(US); v2[k]=v+(1e-5 if hi else -1e-5)
        R2=ranks(v2); f2=sum(R2[x] for x in K[:3])/3; g2=sum(R2[x] for x in K[3:])/3
        need=v-US[k] if hi else US[k]-v
        parts.append(f"第{target}名>{v:.5f}(差{need:+.4f},FINAL {(f2+g2)/2:.3f})")
    print(line + "  ".join(parts[:3]))
