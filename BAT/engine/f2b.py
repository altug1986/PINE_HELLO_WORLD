import numpy as np, pandas as pd, itertools, json, numba as nb
from f2 import *
from met import metrics

@nb.njit(cache=True)
def apply_cb(t, c, want, cbLoss, cbAmt, cbDays, tick):
    n = len(c); real = np.zeros(n, np.int64); pl = np.zeros(n); nt = 0
    pos = 0; pEp = 0.0; pQ = 0.0; cbSt = 0; cbSum = 0.0; seen = 0; until = 0; paused = False; wait = False; last = 0
    for i in range(n):
        while seen < nt:
            if pl[seen] < 0: cbSt += 1; cbSum -= pl[seen]
            else: cbSt = 0; cbSum = 0.0
            seen += 1
        w = want[i]; rw = w
        if paused and t[i] > until: paused = False; cbSt = 0; cbSum = 0.0; wait = True; last = w
        if paused: rw = 0
        elif cbSt >= cbLoss and cbSum >= cbAmt: rw = 0; paused = True; until = t[i] + int(round(cbDays * 86400000))
        elif wait:
            if w != last: wait = False
            else: rw = 0
        real[i] = rw
        if rw != pos:
            if pos != 0:
                x = c[i] - tick * pos; pl[nt] = (x - pEp) * pQ * pos - COMM * pQ * (pEp + x); nt += 1
            if rw != 0: pEp = c[i] + tick * rw; pQ = CASH / c[i]
            pos = rw
    return real

if __name__ == '__main__':
    D = cut(load(5), SPLIT); B = json.load(open('f2_best.json')); order = B['order']; prm = B['prm']
    tg = chain([motor(D, m, prm[m]) for m in order]); m0 = metrics(D, simR(D, tg)); s0 = score(m0)
    K = ['net', 'n', 'pf', 'sharpe', 'p95', 'p98', 'p99', 'mddc', 'upi', 'posM']
    print('BAZ', {k: m0[k] for k in K}, 'skor', round(s0))
    # ardışık 50-işlem blok PF korelasyonu (kesici için ön test)
    R = simR(D, tg); p = R['pnl']; b = [p[i:i+50] for i in range(0, len(p) - 49, 50)]
    pf = np.array([x[x > 0].sum() / max(-x[x < 0].sum(), 1e-9) for x in b]); print('blok PF korelasyonu', round(np.corrcoef(pf[:-1], pf[1:])[0, 1], 3))
    rows = []
    for cl, ca, cd in itertools.product([2, 3, 4], [40, 60, 80, 100], [0.5, 1, 2]):
        r = apply_cb(D['t'], D['c'], tg, cl, ca, cd, TICK); m = metrics(D, simR(D, r))
        rows.append(dict(ek=f'kesici {cl}x {ca}$ {cd}g', **{k: m[k] for k in K}, skor=round(score(m))))
    for n, k, pos in itertools.product([48, 96, 144, 288], [3, 3.5, 4], [1, 2, 3]):
        z2 = dict(n=n, k=k, ex=0.5, sl=8, mb=288); tgs = [motor(D, mm, prm[mm]) for mm in order]; tgs.insert(pos, motor(D, 'Z', z2))
        m = metrics(D, simR(D, chain(tgs))); rows.append(dict(ek=f'Z2 n{n} k{k} sıra{pos}', **{k_: m[k_] for k_ in K}, skor=round(score(m))))
    df = pd.DataFrame(rows); df['Δskor'] = df.skor - round(s0)
    print('kesici pozitif oran', (df[df.ek.str.startswith('kesici')]['Δskor'] > 0).mean())
    print(df.sort_values('skor', ascending=False).head(12).to_string(index=False))
