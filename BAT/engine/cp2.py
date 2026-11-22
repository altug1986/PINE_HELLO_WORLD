import numpy as np, numba as nb, pandas as pd, itertools, json, time
from cp import *

@nb.njit(cache=True)
def layer2(c, core, atr, oi, b24, mode, thr, cSL, cBars, oiX, bX, tick):
    # koşullu ters yön: ters gidiş ≥ eşik VE OI girişten beri ≥ oiX% VE BTC 24s hareket yönü aynı (≥ bX%)
    n = len(c); want = np.zeros(n, np.int64); sh = 0; shEp = 0.0; shOi = np.nan; st = 0; cDir = 0; cPx = 0.0; cBar = 0; prev = 0
    for i in range(n):
        tg = core[i]; evt = tg != prev; prev = tg; w = sh
        th = thr / 100 if mode == 0 else thr * (0.0 if np.isnan(atr[i]) else atr[i]) / 100
        if evt:
            st = 0; w = tg
        elif st == 0:
            w = tg
            if sh != 0 and sh == tg and th > 0 and (c[i] / shEp - 1) * (-sh) >= th:
                okO = oiX < -50 or ((not np.isnan(oi[i])) and (not np.isnan(shOi)) and (oi[i] / shOi - 1) * 100 >= oiX)
                okB = bX < -50 or ((not np.isnan(b24[i])) and b24[i] * (-sh) >= bX)
                if okO and okB:
                    w = -sh; st = 1; cDir = sh; cPx = c[i]; cBar = i
        else:
            g = (c[i] / cPx - 1) * (-cDir)
            if g <= -cSL / 100 or i - cBar >= cBars:
                st = 0; w = tg
            else:
                w = -cDir
        if w != sh:
            sh = w; shEp = c[i] + tick * w if w != 0 else 0.0; shOi = oi[i]
        want[i] = w
    return want

@nb.njit(cache=True)
def g_conf(c, sig, hold, sl, k, r):
    # ön keşif + teyit: sinyal varken giriş ancak son k barın dibinden (long) / tepesinden (short) r% dönüşle
    L = len(c); tg = np.zeros(L, np.int64); v = 0; e = 0.0; vb = 0
    for i in range(L):
        if v != 0:
            if (c[i] / e - 1) * v <= -sl / 100 or i - vb >= hold: v = 0
        s = sig[i]
        if s != 0:
            if v == s: vb = i
            else:
                lo = c[i]; hi = c[i]
                for j in range(max(0, i - k), i + 1):
                    lo = min(lo, c[j]); hi = max(hi, c[j])
                ok = (s > 0 and c[i] >= lo * (1 + r / 100)) or (s < 0 and c[i] <= hi * (1 - r / 100)) or k == 0
                if ok: v = s; e = c[i]; vb = i
        tg[i] = v
    return tg

def aux(D, lag=0):
    o = np.full(len(D['c']), np.nan); k = 1 + lag; o[k:] = D['oi'][:-k]; x = D['xc']
    b24 = (x / np.r_[np.full(288, np.nan), x[:-288]] - 1) * 100
    return o, b24

if __name__ == '__main__':
    DI = cut(load(5), SPLIT); FI = feats(DI); core = mot2(DI, FI, CORE); o, b24 = aux(DI)
    b0 = metrics(DI, simR(DI, core)); print('IS BAZ', {k: b0[k] for k in KS})
    rows = []
    for (mode, thr), cSL, cBars, oiX, bX in itertools.product([(1, 3), (1, 4), (1, 6), (0, 6), (0, 10)], [2, 4, 6], [96, 288, 576], [-99, 0, 2, 5], [-99, 0, 1, 2]):
        R = simR(DI, layer2(DI['c'], core, DI['atr1h'], o, b24, mode, thr, cSL, cBars, oiX, bX, TICK)); m = metrics(DI, R)
        rows.append(dict(mode=mode, thr=thr, cSL=cSL, cBars=cBars, oiX=oiX, bX=bX, **{k: m[k] for k in KS}, sc=sc(m)))
    df = pd.DataFrame(rows); df.to_pickle('cp2_IS.pkl'); pd.set_option('display.width', 250)
    print('A koşullu ters: bazı geçen', ((df.net >= b0['net'] * 1.06) & (df.p98 <= b0['p98'] * 1.1)).sum(), '/', len(df), '| net>baz', (df.net > b0['net']).sum())
    print(df.sort_values('sc', ascending=False).head(6).to_string(index=False))
    s = sig2(FI, DI, CORE); rows = []
    for k, r in itertools.product([3, 6, 12, 24, 48], [0.3, 0.5, 1, 1.5, 2, 3]):
        m = metrics(DI, simR(DI, g_conf(DI['c'], s, 648, 3, k, r))); rows.append(dict(k=k, r=r, **{kk: m[kk] for kk in KS}, sc=sc(m)))
    d2 = pd.DataFrame(rows); print('D teyitli giriş: net>baz', (d2.net > b0['net']).sum(), '/', len(d2)); print(d2.sort_values('sc', ascending=False).head(6).to_string(index=False))
