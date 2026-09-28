"""Predict the official RANK of a candidate from its val metrics.

Calibration anchor: submission #19 pb_p50_b45 (4-member base + Difix t199, person 0.50 / bg 0.45) — the same
recipe whose val numbers we have in fixtest (which holds the 4-member base render). val->test offsets are
taken from that one matched pair and applied additively to any other candidate measured on the same dumps.

Usage: rank_predict.py <metrics_log_001_1> <metrics_log_012_0> [--live]
The metrics logs are the output of work/metrics_both.py (variant, FULL psnr/ssim/alex/vgg, FG psnr/ssim/alex/vgg).
"""
import sys, json, ssl, urllib.request, argparse, re

ANCHOR = 'pb_p050_b045'   # val variant name
ANCHOR_TEST = dict(psnr=25.9436, ssim=0.9185477, lpips=0.1864870,
                   fg_psnr=24.2482, fg_ssim=0.8324715, fg_lpips=0.2771606)

def parse(path):
    out = {}
    for line in open(path):
        m = re.match(r'^(\S+)\s+([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s+([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)', line)
        if m:
            v = m.group(1); f = [float(x) for x in m.groups()[1:]]
            out[v] = dict(psnr=f[0], ssim=f[1], alex=f[2], vgg=f[3], fg_psnr=f[4], fg_ssim=f[5], fg_alex=f[6], fg_vgg=f[7])
    return out

ap = argparse.ArgumentParser(); ap.add_argument('logs', nargs=2); ap.add_argument('--lpips', default='alex', choices=['alex', 'vgg'])
a = ap.parse_args()
A, B = parse(a.logs[0]), parse(a.logs[1])
common = [v for v in A if v in B]
if ANCHOR not in common:
    sys.exit(f'anchor {ANCHOR} missing from both logs (have: {sorted(common)})')

L = a.lpips
def valmean(v): return {k: (A[v][k] + B[v][k]) / 2 for k in A[v]}
anc = valmean(ANCHOR)
OFF = dict(psnr=ANCHOR_TEST['psnr']-anc['psnr'], ssim=ANCHOR_TEST['ssim']-anc['ssim'],
           lpips=ANCHOR_TEST['lpips']-anc[L], fg_psnr=ANCHOR_TEST['fg_psnr']-anc['fg_psnr'],
           fg_ssim=ANCHOR_TEST['fg_ssim']-anc['fg_ssim'], fg_lpips=ANCHOR_TEST['fg_lpips']-anc['fg_'+L])

ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
lb = json.load(urllib.request.urlopen('https://8.136.221.94/api/leaderboard?track=sparse_views', context=ctx, timeout=30))
OPP = {e['display_name']: e['metrics'] for e in lb
       if e['display_name'] != 'Doreen071' and isinstance(e['metrics'], dict) and e['metrics'].get('foreground_psnr') is not None}

def rank(val, others, lower): return sorted(others + [val], reverse=not lower).index(val) + 1
def score(m):
    o = list(OPP.values())
    rf = [rank(m['psnr'], [x['psnr'] for x in o], False), rank(m['ssim'], [x['ssim'] for x in o], False), rank(m['lpips'], [x['lpips'] for x in o], True)]
    rg = [rank(m['fg_psnr'], [x['foreground_psnr'] for x in o], False), rank(m['fg_ssim'], [x['foreground_ssim'] for x in o], False), rank(m['fg_lpips'], [x['foreground_lpips'] for x in o], True)]
    return rf, rg, (sum(rf)/3 + sum(rg)/3)/2

print(f'val->test offsets (LPIPS={L}): ' + ' '.join(f'{k}{v:+.4f}' for k, v in OFF.items()))
print(f'opponents: {", ".join(OPP)}')
print(f"\n{'variant':22s} {'predicted test FULL':>26s} {'predicted test FG':>26s}  {'FULL':>9s} {'FG':>9s} {'RANK':>6s}")
rows = []
for v in common:
    vm = valmean(v)
    p = dict(psnr=vm['psnr']+OFF['psnr'], ssim=vm['ssim']+OFF['ssim'], lpips=vm[L]+OFF['lpips'],
             fg_psnr=vm['fg_psnr']+OFF['fg_psnr'], fg_ssim=vm['fg_ssim']+OFF['fg_ssim'], fg_lpips=vm['fg_'+L]+OFF['fg_lpips'])
    rf, rg, fin = score(p); rows.append((fin, v, p, rf, rg))
for fin, v, p, rf, rg in sorted(rows):
    print(f"{v:22s} {p['psnr']:8.3f}/{p['ssim']:.5f}/{p['lpips']:.5f} {p['fg_psnr']:8.3f}/{p['fg_ssim']:.5f}/{p['fg_lpips']:.5f}  {str(rf):>9s} {str(rg):>9s} {fin:6.3f}")
