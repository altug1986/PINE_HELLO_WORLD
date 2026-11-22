import numpy as np, pandas as pd, v55
from numba import njit
from v55 import D, t, n, evalp, run, simR
from search57 import SCs
from rg55 import ADX60, AGREE
from astre_test import astre, TS, q0
from mtm30 import m30
HMS = 3600000
GLb, GSb = astre(-300000); S1 = t >= (pd.Timestamp('2026-04-01') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
GL0, GS0 = GLb & S1, GSb & S1                                   # TV'deki V6.3: balina kapısı 2026-04 sonrası
SC = SCs[2.5]
@njit
def st_dir(h, l, c, ln, fac):                                   # TV ta.supertrend; +1 yukarı / −1 aşağı
    m = len(c); d = np.zeros(m, np.int64); atr = np.full(m, np.nan); tr = np.zeros(m)
    for i in range(m): tr[i] = h[i] - l[i] if i == 0 else max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    s = 0.0
    for i in range(m):
        if i < ln - 1: s += tr[i]
        elif i == ln - 1: s += tr[i]; atr[i] = s / ln
        else: atr[i] = (atr[i - 1] * (ln - 1) + tr[i]) / ln
    up = 0.0; lo = 0.0; stv = 0.0; dr = 1
    for i in range(m):
        if np.isnan(atr[i]): d[i] = 0; continue
        src = (h[i] + l[i]) / 2; U = src + fac * atr[i]; L = src - fac * atr[i]
        pU = up; pL = lo
        if i > 0 and not np.isnan(atr[i - 1]):
            L = L if (L > pL or c[i - 1] < pL) else pL; U = U if (U < pU or c[i - 1] > pU) else pU
            if stv == pU: dr = -1 if c[i] > U else 1
            else: dr = 1 if c[i] < L else -1
        else: dr = 1
        stv = L if dr == -1 else U; up = U; lo = L; d[i] = -dr
    return d
def st5(hrs, ln, fac, lag=1):                                   # HTF kapanmış mum yönü → 5 dk (lookahead_on + [lag])
    k = int(round(hrs * HMS)); g = t // k
    df = pd.DataFrame(dict(g=g, h=D['h'], l=D['l'], c=D['c'])).groupby('g').agg(h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    d = st_dir(df.h.to_numpy(), df.l.to_numpy(), df.c.to_numpy(), ln, fac); gi = df.index.to_numpy()
    j = np.searchsorted(gi, g) - lag; return np.where(j >= 0, d[np.clip(j, 0, None)], 0)
@njit
def vote_st(sc, adx, ag, st, aT, on, onT, onCS, onW, offW):
    m = len(sc); tg = np.zeros(m, np.int64); pos = 0
    for i in range(m):
        s = sc[i]; want = pos; off = offW if (pos != 0 and st[i] == pos) else 0.0
        if pos != 0 and s * pos < off: want = 0
        oL = onT if (ag[i] == -1 and adx[i] >= aT) else on
        oS = onT if (ag[i] == 1 and adx[i] >= aT) else on
        if st[i] == 1:
            if oL == on: oL = onW
            oS = max(oS, onCS)
        elif st[i] == -1:
            if oS == on: oS = onW
            oL = max(oL, onCS)
        if s >= oL: want = 1
        elif s <= -oS: want = -1
        pos = want; tg[i] = pos
    return tg
TG0 = v55.TG.copy()
def ev(st, onCS=0.04, onW=0.04, offW=0.0, filt=False, full=False, lab=''):
    v55.TG = TG0 if st is None else vote_st(SC, ADX60, AGREE, st, 20.0, 0.04, 0.15, onCS, onW, offW)
    gl, gs = (GLb, GSb) if full else (GL0, GS0)
    if filt: gl = gl | (st == -1); gs = gs | (st == 1)
    v55.GL = gl.astype(np.int64); v55.GS = gs.astype(np.int64); q = dict(q0, **TS); e = evalp(q); m = m30(simR(D, run(q)[0]))
    return dict(t=lab, net=e['net'], n=e['n'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], DD=m['MaxDD 30dk mum içi'], Cal=m['Calmar 30dk'], P=m['DD P95/P98/P99 30dk'], SS=m['Sharpe/Sortino 30dk'])
@njit
def vote_st2(sc, adx, ag, stH, stT, aT, on, onT, onCS, offW):   # stH: tutma için, stT: ters eşik için
    m = len(sc); tg = np.zeros(m, np.int64); pos = 0
    for i in range(m):
        s = sc[i]; want = pos; off = offW if (pos != 0 and stH[i] == pos) else 0.0
        if pos != 0 and s * pos < off: want = 0
        oL = onT if (ag[i] == -1 and adx[i] >= aT) else on
        oS = onT if (ag[i] == 1 and adx[i] >= aT) else on
        if stT[i] == 1: oS = max(oS, onCS)
        elif stT[i] == -1: oL = max(oL, onCS)
        if s >= oL: want = 1
        elif s <= -oS: want = -1
        pos = want; tg[i] = pos
    return tg
def ev2(stH, stT, onCS, full=False, lab=''):
    v55.TG = vote_st2(SC, ADX60, AGREE, stH, stT, 20.0, 0.04, 0.15, onCS, -0.15)
    gl, gs = (GLb, GSb) if full else (GL0, GS0); v55.GL = gl.astype(np.int64); v55.GS = gs.astype(np.int64)
    q = dict(q0, **TS); e = evalp(q); m = m30(simR(D, run(q)[0]))
    return dict(t=lab, net=e['net'], n=e['n'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], DD=m['MaxDD 30dk mum içi'], Cal=m['Calmar 30dk'], P=m['DD P95/P98/P99 30dk'], SS=m['Sharpe/Sortino 30dk'])
