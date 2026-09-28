"""Gap analysis: for each metric, how much do we need to gain to move up one rank, and what does it do to the final score.
Also runs 'what-if' scenarios, including the effect our own improvement has on opponents' ranks."""
import json, ssl, urllib.request, itertools

ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
lb = json.load(urllib.request.urlopen('https://8.136.221.94/api/leaderboard?track=sparse_views', context=ctx, timeout=30))
T = {e['display_name']: dict(e['metrics']) for e in lb
     if isinstance(e['metrics'], dict) and e['metrics'].get('foreground_psnr') is not None}
# our LATEST VALID submission is what counts, not the leaderboard's best-ever row
OURS = dict(psnr=25.943567, ssim=0.9185477, lpips=0.1864870,
            foreground_psnr=24.248189, foreground_ssim=0.8324715, foreground_lpips=0.2771606)
T['Doreen071'] = OURS
KEYS = [('psnr', 0), ('ssim', 0), ('lpips', 1), ('foreground_psnr', 0), ('foreground_ssim', 0), ('foreground_lpips', 1)]
LBL = ['FULL-PSNR', 'FULL-SSIM', 'FULL-LPIPS', 'FG-PSNR', 'FG-SSIM', 'FG-LPIPS']

def ranks(table, who):
    out = []
    for k, lower in KEYS:
        vals = sorted((t[k] for t in table.values()), reverse=not lower)
        out.append(vals.index(table[who][k]) + 1)
    return out
def final(table, who):
    r = ranks(table, who); return (sum(r[:3])/3 + sum(r[3:])/3)/2

print('=== CURRENT STANDINGS (using our LATEST VALID submission) ===')
rows = sorted(((final(T, n), n, ranks(T, n)) for n in T))
for f, n, r in rows:
    print(f'  {f:6.3f}  {n:<14} FULL{r[:3]} FG{r[3:]}')

print('\n=== PER-METRIC GAP: what we need to move up ONE rank ===')
for i, (k, lower) in enumerate(KEYS):
    mine = OURS[k]
    better = sorted([(n, t[k]) for n, t in T.items() if n != 'Doreen071' and ((t[k] < mine) if lower else (t[k] > mine))],
                    key=lambda x: x[1], reverse=not lower)
    cur = ranks(T, 'Doreen071')[i]
    if not better:
        print(f'  {LBL[i]:<11} rank {cur}  ALREADY FIRST'); continue
    tgt_name, tgt = better[-1]          # the one just ahead of us
    need = (mine - tgt) if lower else (tgt - mine)
    print(f'  {LBL[i]:<11} rank {cur} -> {cur-1}: beat {tgt_name:<12} {tgt:.5f}   need {"-" if lower else "+"}{abs(need):.5f}  (we are {mine:.5f})')

print('\n=== WHAT-IF SCENARIOS (epsilon = just barely beat the target) ===')
EPS = 1e-4
def scenario(name, changes):
    t = {n: dict(v) for n, v in T.items()}
    for k, v in changes.items(): t['Doreen071'][k] = v
    r = ranks(t, 'Doreen071'); f = final(t, 'Doreen071')
    board = sorted(((final(t, n), n) for n in t))
    winner = board[0]
    print(f'  {name}')
    print(f'     us: FULL{r[:3]} FG{r[3:]} -> {f:.3f}   | board top-3: ' +
          '  '.join(f'{n} {s:.3f}' for s, n in board[:3]) + f'   [{"WE WIN" if winner[1]=="Doreen071" else "led by "+winner[1]}]')

beat_ssim  = T['YunqiGao123!']['ssim'] + EPS
beat_lpips = T['shengqi']['lpips'] - EPS
beat_fgssim = T['shengqi']['foreground_ssim'] + EPS
beat_fglpips = T['mmm']['foreground_lpips'] - EPS
beat_fgpsnr = T['shengqi']['foreground_psnr'] + EPS

scenario('A. FULL-SSIM only (+0.0017)', {'ssim': beat_ssim})
scenario('B. FULL-LPIPS only (-0.0046)', {'lpips': beat_lpips})
scenario('C. FULL-SSIM + FULL-LPIPS', {'ssim': beat_ssim, 'lpips': beat_lpips})
scenario('D. C + FG-LPIPS (beat mmm)', {'ssim': beat_ssim, 'lpips': beat_lpips, 'foreground_lpips': beat_fglpips})
scenario('E. D + FG-SSIM (beat shengqi)', {'ssim': beat_ssim, 'lpips': beat_lpips,
                                           'foreground_lpips': beat_fglpips, 'foreground_ssim': beat_fgssim})
scenario('F. E + FG-PSNR (beat shengqi, +1.3dB)', {'ssim': beat_ssim, 'lpips': beat_lpips,
                                                   'foreground_lpips': beat_fglpips, 'foreground_ssim': beat_fgssim,
                                                   'foreground_psnr': beat_fgpsnr})
