"""Pre-compute our optimal response for plausible opponent moves, so a change on the board
triggers an immediate decision instead of a fresh analysis.
Our achievable configs come from the measured person/bg Difix sweep (val->test calibrated)."""
OURS = {   # label: (FULL psnr,ssim,lpips, FG psnr,ssim,lpips)
 'p0.0/b0.30 (light)':   (25.879,0.9233,0.1980, 24.391,0.8393,0.3500),
 'p0.2/b0.30':           (25.894,0.9234,0.1952, 24.464,0.8401,0.3283),
 'p0.3/b0.45':           (25.984,0.9205,0.1778, 24.479,0.8374,0.3056),
 'p0.5/b0.45 (planned)': (25.970,0.9201,0.1757, 24.418,0.8350,0.2899),
 'p0.7/b0.45':           (25.937,0.9196,0.1747, 24.275,0.8310,0.2823),
 'p0.5/b0.55':           (25.995,0.9176,0.1683, 24.398,0.8329,0.2850),
 'p0.5/b0.70':           (25.979,0.9130,0.1624, 24.336,0.8291,0.2812),
}
BASE = {
 'shengqi': (27.682,0.9180912,0.1819135, 25.551,0.8393987,0.2949982),
 'mmm':     (24.979,0.9111418,0.2788681, 24.192,0.8292191,0.2658243),
 'mingzai': (23.732,0.9113840,0.2574871, 23.035,0.8247841,0.4193558),
}
SCEN = {
 'current board': {},
 'shengqi pushes LPIPS (0.17/0.27)': {'shengqi': (27.682,0.9180912,0.1700, 25.551,0.8393987,0.2700)},
 'shengqi pushes SSIM (0.925)':      {'shengqi': (27.682,0.9250000,0.1819135, 25.551,0.8450000,0.2949982)},
 'mmm improves PSNR (+1.5dB)':       {'mmm':     (26.479,0.9200000,0.2600000, 25.192,0.8350000,0.2500000)},
}
def rank(v, others, lower): return sorted(others+[v], reverse=not lower).index(v)+1
def evaluate(us, opp):
    o=list(opp.values())
    rf=[rank(us[0],[x[0] for x in o],False), rank(us[1],[x[1] for x in o],False), rank(us[2],[x[2] for x in o],True)]
    rg=[rank(us[3],[x[3] for x in o],False), rank(us[4],[x[4] for x in o],False), rank(us[5],[x[5] for x in o],True)]
    ours=(sum(rf)/3+sum(rg)/3)/2
    best_opp=1e9; who=None
    for n,m in opp.items():
        oth=[us]+[x for k,x in opp.items() if k!=n]
        srf=[rank(m[0],[x[0] for x in oth],False), rank(m[1],[x[1] for x in oth],False), rank(m[2],[x[2] for x in oth],True)]
        srg=[rank(m[3],[x[3] for x in oth],False), rank(m[4],[x[4] for x in oth],False), rank(m[5],[x[5] for x in oth],True)]
        s=(sum(srf)/3+sum(srg)/3)/2
        if s<best_opp: best_opp, who = s, n
    return ours, best_opp, who
for sname, override in SCEN.items():
    opp = dict(BASE); opp.update(override)
    rows=[(evaluate(v,opp)[0],)+evaluate(v,opp)[1:]+(k,) for k,v in OURS.items()]
    rows.sort()
    best=rows[0]
    verdict = 'WIN' if best[0] < best[1]-1e-9 else ('TIE' if abs(best[0]-best[1])<1e-9 else f'2nd behind {best[2]}')
    print(f'{sname:36s} best-for-us: {best[3]:<22s} ours={best[0]:.3f} vs {best[2]}={best[1]:.3f}  -> {verdict}')
