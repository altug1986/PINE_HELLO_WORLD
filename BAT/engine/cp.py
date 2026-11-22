import numpy as np, numba as nb, pandas as pd, itertools, json, sys, time
from sysB import *
from met import metrics

@nb.njit(cache=True)
def layer(c, core, atr, mode, thr, cSL, cBars, back, tick):
    # gölge pozisyon + ters yön (trend takip) katmanı
    # mode 0: eşik sabit %, 1: eşik = thr × 1H ATR%
    n = len(c); want = np.zeros(n, np.int64); sh = 0; shEp = 0.0; st = 0; cDir = 0; cPx = 0.0; cBar = 0; prev = 0
    for i in range(n):
        tg = core[i]; evt = tg != prev; prev = tg; w = sh
        th = thr / 100 if mode == 0 else thr * (0.0 if np.isnan(atr[i]) else atr[i]) / 100
        if evt:
            st = 0; w = tg
        elif st == 0:
            w = tg
            if sh != 0 and sh == tg and th > 0 and (c[i] / shEp - 1) * (-sh) >= th:
                w = -sh; st = 1; cDir = sh; cPx = c[i]; cBar = i
        else:
            g = (c[i] / cPx - 1) * (-cDir)
            if g <= -cSL / 100 or i - cBar >= cBars:
                st = 0; w = tg if back == 0 else 0
                if back == 1: st = 2
            else:
                w = -cDir
        if st == 2:
            w = 0  # flat bekle: çekirdek yeni olay üretene kadar
        if w != sh:
            sh = w; shEp = c[i] + tick * w if w != 0 else 0.0
        want[i] = w
    return want

def mae_of(D, R):
    w = np.array([(np.min(D['l'][a + 1:b + 1]) / e - 1) * 100 if s > 0 else -(np.max(D['h'][a + 1:b + 1]) / e - 1) * 100
                  for a, b, s, e in zip(R['eb'], R['xb'], R['dir'], R['ep']) if b > a])
    return np.percentile(w, [10, 5, 1]).round(1), round(w.min(), 1), int((w <= -10).sum())

CORE = dict(fam='PXA', H=288, aL=3.5, aS=5.5, hold=648, sl=3)
def runL(D, F, core_s, L):
    core = mot2(D, F, core_s)
    return simR(D, layer(D['c'], core, D['atr1h'], L['mode'], L['thr'], L['cSL'], L['cBars'], L['back'], TICK))

if __name__ == '__main__':
    DI = cut(load(5), SPLIT); FI = feats(DI); rows = []; t0 = time.time()
    b0 = metrics(DI, simR(DI, mot2(DI, FI, CORE))); print('IS BAZ V1.0', {k: b0[k] for k in KS})
    for mode, thrs in [(0, [2, 3, 4, 5, 6, 8, 10]), (1, [2, 3, 4, 5, 6])]:
        for thr, cSL, cBars, back in itertools.product(thrs, [1, 2, 3, 4, 6], [48, 96, 144, 288, 576], [0, 1]):
            L = dict(mode=mode, thr=thr, cSL=cSL, cBars=cBars, back=back); R = runL(DI, FI, CORE, L); m = metrics(DI, R)
            rows.append(dict(**L, **{k: m[k] for k in KS}, sortino=m['sortino'], sc=sc(m)))
    df = pd.DataFrame(rows); df.to_pickle('cp_IS.pkl'); print(f'{time.time()-t0:.0f}s', len(df))
    print('bazı geçen (net≥+%6 & P98≤baz & Sharpe↑)', ((df.net >= b0['net'] * 1.06) & (df.p98 <= b0['p98']) & (df.sharpe > b0['sharpe'])).mean().round(2))
    pd.set_option('display.width', 250); print(df.sort_values('sc', ascending=False).head(12).to_string(index=False))
    print(df.sort_values('net', ascending=False).head(5).to_string(index=False))
