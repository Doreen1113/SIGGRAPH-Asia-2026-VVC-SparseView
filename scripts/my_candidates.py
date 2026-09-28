"""Rank every package we have actually had scored, under the live board. This is how the final submission gets
chosen: by computed Final Rank from the six real metrics, never by PSNR alone."""
import json, ssl, urllib.request, datetime, sys
K = ['psnr','ssim','lpips','foreground_psnr','foreground_ssim','foreground_lpips']
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
TOKEN = sys.argv[1]
def get(u, tok=False):
    r = urllib.request.Request(u, headers={'Authorization': f'Bearer {TOKEN}'} if tok else {})
    return json.load(urllib.request.urlopen(r, context=ctx, timeout=30))
rows = get('https://8.136.221.94/api/leaderboard?track=sparse_views')
rows = rows if isinstance(rows, list) else rows.get('entries', [])
teams = {}
for r in rows:                      # the baseline row lacks foreground metrics; skip anything incomplete
    m = r.get('metrics') or {}
    if all(k in m for k in K): teams[r['username']] = {k: float(m[k]) for k in K}
teams.pop('doreen071', None)
d = get('https://8.136.221.94/api/submissions?track=sparse_views', True)
subs = d if isinstance(d, list) else d.get('submissions', d.get('items', []))
def final(v):
    t = dict(teams); t['US'] = v; rk = {}
    for k in K:
        hi = 'lpips' not in k
        order = sorted(t.items(), key=lambda kv: -kv[1][k] if hi else kv[1][k])
        rk[k] = [n for n, _ in order].index('US') + 1
    full = sum(rk[k] for k in K[:3]) / 3; fg = sum(rk[k] for k in K[3:]) / 3
    return (full + fg) / 2, full, fg, [rk[k] for k in K]
out = []
for s in subs:
    m = s.get('metrics') or {}
    if s.get('status') != 'succeeded' or not all(k in m for k in K): continue
    v = {k: float(m[k]) for k in K}
    f, fu, fg, rk = final(v)
    out.append((f, fu, fg, rk, s['filename'], datetime.datetime.fromtimestamp(s['created_at']).strftime('%m-%d %H:%M'), v))
out.sort(key=lambda x: x[0])
print(f"{'FINAL':>6} {'FULL':>5} {'FG':>5}  {'ranks':<22} {'when':<12} package")
for f, fu, fg, rk, fn, when, v in out:
    print(f"{f:6.3f} {fu:5.3f} {fg:5.3f}  {str(rk[:3])+str(rk[3:]):<22} {when:<12} {fn}")
print(f"\n最佳 = {out[0][4]}  (FINAL {out[0][0]:.3f})")
print("  ", {k: round(out[0][6][k], 5) for k in K})
