"""Simulate the whole leaderboard: our metrics change other teams' ranks too, so the only way to know our
place is to recompute every team's Final Rank. Usage: board_sim.py [psnr ssim lpips fgpsnr fgssim fglpips]"""
import json, ssl, sys, urllib.request
K = ['psnr','ssim','lpips','foreground_psnr','foreground_ssim','foreground_lpips']
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
lb = json.load(urllib.request.urlopen('https://8.136.221.94/api/leaderboard?track=sparse_views', context=ctx, timeout=30))
rows = lb if isinstance(lb, list) else lb.get('entries', lb.get('items', []))
teams = {}
for r in rows:
    n = r.get('team_name') or r.get('username') or r.get('display_name'); m = r.get('metrics') or r
    try: teams[n] = {k: float(m[k]) for k in K}
    except Exception: pass
teams.pop('doreen071', None)

def standings(mine):
    t = dict(teams); t['US'] = dict(zip(K, mine))
    rk = {}
    for k in K:
        hi = 'lpips' not in k
        order = sorted(t.items(), key=lambda kv: -kv[1][k] if hi else kv[1][k])
        for i, (n, _) in enumerate(order): rk.setdefault(n, {})[k] = i + 1
    out = []
    for n, r in rk.items():
        full = sum(r[k] for k in K[:3]) / 3; fg = sum(r[k] for k in K[3:]) / 3
        out.append((( full + fg) / 2, full, fg, n, [r[k] for k in K]))
    return sorted(out)

if len(sys.argv) == 7:
    mine = [float(x) for x in sys.argv[1:]]
else:
    mine = [26.071964, 0.9203802, 0.1817023, 24.4168756, 0.8271100, 0.2438235]  # FGE_p085, the live score
for i, (fin, full, fg, n, r) in enumerate(standings(mine), 1):
    print(f"{i}. {n:<14} FINAL {fin:.3f}  FULL {full:.3f} {r[:3]}  FG {fg:.3f} {r[3:]}")
