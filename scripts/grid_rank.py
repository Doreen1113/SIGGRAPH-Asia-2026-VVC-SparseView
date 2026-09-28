"""Rank every val candidate against the live board, using offsets calibrated from our four scored submissions.
Usage: grid_rank.py <metrics_log> [<metrics_log> ...]
Each log line: "<variant>  psnr/ssim/alex/vgg   fg_psnr/fg_ssim/fg_alex/fg_vgg"
The anchor variant (default sub070) must appear in the first log; its real scored values pin the offsets."""
import sys, re, json, ssl, urllib.request

ANCHOR = 'sub070'                       # val analogue of the scored SIX_p070 package
ANCHOR_REAL = (26.0309, 0.92066, 0.18312, 24.2483, 0.83203, 0.26493)
ERR = dict(ssim=0.002, lpips=0.011)     # historical prediction error, used to normalise margins

ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
lb = json.load(urllib.request.urlopen('https://8.136.221.94/api/leaderboard?track=sparse_views', context=ctx, timeout=30))
OPP = {e['display_name']: e['metrics'] for e in lb
       if e['display_name'] != 'Doreen071' and isinstance(e['metrics'], dict)
       and e['metrics'].get('foreground_psnr') is not None}
K = [('psnr', 0), ('ssim', 0), ('lpips', 1), ('foreground_psnr', 0), ('foreground_ssim', 0), ('foreground_lpips', 1)]

def parse(paths):
    out = {}
    for p in paths:
        for line in open(p):
            m = re.match(r'^(\S+)\s+([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s+([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)', line)
            if m:
                v = m.group(1); f = [float(x) for x in m.groups()[1:]]
                out[v] = (f[0], f[1], f[2], f[4], f[5], f[6])   # psnr,ssim,alex | fg_psnr,fg_ssim,fg_alex
    return out

def ranks(me):
    t = {n: dict(v) for n, v in OPP.items()}; t['Doreen071'] = me
    return [sorted((x[k] for x in t.values()), reverse=not lo).index(t['Doreen071'][k]) + 1 for k, lo in K]

def worst_margin(me):
    """Normalised distance to the nearest threshold we are currently on the right side of."""
    ms = []
    for k, lo in K:
        vals = sorted((x[k] for x in OPP.values()), reverse=not lo)
        better = [v for v in vals if (v > me[k]) == (not lo)]   # opponents ahead of us on this metric
        worse = [v for v in vals if (v > me[k]) != (not lo)]
        if not worse: continue                                   # last place on this metric: no margin to defend
        nearest = worse[0] if lo else worse[0]
        d = abs(me[k] - nearest)
        scale = ERR['lpips'] if 'lpips' in k else (ERR['ssim'] if 'ssim' in k else 0.5)
        ms.append(d / scale)
    return min(ms) if ms else 0.0

cands = parse(sys.argv[1:])
if ANCHOR not in cands:
    sys.exit(f'anchor {ANCHOR} not found in the given logs')
a = cands[ANCHOR]
off = [ANCHOR_REAL[i] - a[i] for i in range(6)]
rows = []
for v, f in cands.items():
    me = dict(zip(('psnr', 'ssim', 'lpips', 'foreground_psnr', 'foreground_ssim', 'foreground_lpips'),
                  [f[i] + off[i] for i in range(6)]))
    r = ranks(me); fin = (sum(r[:3]) / 3 + sum(r[3:]) / 3) / 2
    rows.append((fin, -worst_margin(me), v, r, me))
rows.sort()
print(f'{"variant":22s} {"rank":>6s} {"margin":>7s}  FULL psnr/ssim/lpips        FG psnr/ssim/lpips')
for fin, nm, v, r, me in rows[:20]:
    print(f'{v:22s} {fin:6.3f} {-nm:6.2f}s  {me["psnr"]:.3f}/{me["ssim"]:.5f}/{me["lpips"]:.5f}  '
          f'{me["foreground_psnr"]:.3f}/{me["foreground_ssim"]:.5f}/{me["foreground_lpips"]:.5f}  {r[:3]}{r[3:]}')
