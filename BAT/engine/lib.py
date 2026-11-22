import numpy as np, numba as nb, pandas as pd, itertools, sys, time
from eng import *
from met import SPLIT

@nb.njit(cache=True)
def sim(c, tg, tick):
    n = len(c); eb = np.zeros(n, np.int64); xb = np.zeros(n, np.int64); dr = np.zeros(n, np.int64)
    ep = np.zeros(n); xp = np.zeros(n); qt = np.zeros(n); pl = np.zeros(n); nt = 0
    pos = 0; pEp = 0.0; pQ = 0.0; pEb = 0
    for i in range(n):
        rw = tg[i]
        if rw != pos:
            if pos != 0:
                x = c[i] - tick * pos
                eb[nt] = pEb; xb[nt] = i; dr[nt] = pos; ep[nt] = pEp; xp[nt] = x; qt[nt] = pQ
                pl[nt] = (x - pEp) * pQ * pos - COMM * pQ * (pEp + x); nt += 1
            if rw != 0:
                pEp = c[i] + tick * rw; pQ = CASH / c[i]; pEb = i
            pos = rw
    return eb[:nt], xb[:nt], dr[:nt], ep[:nt], xp[:nt], qt[:nt], pl[:nt], pos, pEb, pEp, pQ

def simR(D, tg):
    r = sim(D['c'], tg.astype(np.int64), TICK)
    return dict(tgt=tg, eb=r[0], xb=r[1], dir=r[2], ep=r[3], xp=r[4], q=r[5], pnl=r[6], open=(r[7], r[8], r[9], r[10]))

@nb.njit(cache=True)
def g_mom(c, adx5, adx30, n, x, a30, rise, hold):
    L = len(c); tg = np.zeros(L, np.int64); v = 0; vb = 0
    for i in range(max(n, rise), L):
        g = True
        if a30 > 0: g = (not np.isnan(adx30[i])) and adx30[i] > a30
        if g and rise > 0: g = (not np.isnan(adx5[i])) and (not np.isnan(adx5[i - rise])) and adx5[i] > adx5[i - rise]
        if g:
            m = (c[i] / c[i - n] - 1) * 100; can = v == 0 or i - vb >= hold
            if m >= x and v <= 0 and can: v = 1; vb = i
            elif m <= -x and v >= 0 and can: v = -1; vb = i
        tg[i] = v
    return tg

@nb.njit(cache=True)
def g_don(c, hh, ll, adx30, a30, hold):
    L = len(c); tg = np.zeros(L, np.int64); v = 0; vb = 0
    for i in range(L):
        g = True
        if a30 > 0: g = (not np.isnan(adx30[i])) and adx30[i] > a30
        can = v == 0 or i - vb >= hold
        if g and not np.isnan(hh[i]):
            if c[i] > hh[i] and v <= 0 and can: v = 1; vb = i
            elif c[i] < ll[i] and v >= 0 and can: v = -1; vb = i
        tg[i] = v
    return tg

@nb.njit(cache=True)
def g_osc(c, x, lo, hi, exL, exS, sl, mb):
    # mean-reversion on oscillator x: x<lo long, x>hi short; exit long when x>exL, short when x<exS; close stop sl%; max bars
    L = len(c); tg = np.zeros(L, np.int64); v = 0; e = 0.0; vb = 0
    for i in range(L):
        xi = x[i]
        if np.isnan(xi): tg[i] = v; continue
        if v != 0:
            r = (c[i] / e - 1) * v
            if r <= -sl / 100 or i - vb >= mb or (v > 0 and xi > exL) or (v < 0 and xi < exS): v = 0
        else:
            if xi < lo: v = 1; e = c[i]; vb = i
            elif xi > hi: v = -1; e = c[i]; vb = i
        tg[i] = v
    return tg

@nb.njit(cache=True)
def g_btc(c, xc, n, x, hold, lagreq):
    L = len(c); tg = np.zeros(L, np.int64); v = 0; vb = 0
    for i in range(n, L):
        if v != 0 and i - vb >= hold: v = 0
        if not np.isnan(xc[i]) and not np.isnan(xc[i - n]):
            br = (xc[i] / xc[i - n] - 1) * 100; ar = (c[i] / c[i - n] - 1) * 100
            if br >= x and (lagreq == 0 or ar < br): v = 1; vb = i
            elif br <= -x and (lagreq == 0 or ar > br): v = -1; vb = i
        tg[i] = v
    return tg

def summ(R):
    p = R['pnl']; gp = p[p > 0].sum(); gl = -p[p < 0].sum(); cp = np.r_[0, np.cumsum(p)]
    return dict(net=round(p.sum()), n=len(p), pf=round(gp / max(gl, 1e-9), 3), mdd=round((np.maximum.accumulate(cp) - cp).max()))

def cut(D, end):
    k = np.searchsorted(D['t'], end); return {a: (v[:k] if isinstance(v, np.ndarray) and len(v) == len(D['t']) else v) for a, v in D.items()}

def zsc(c, n):
    s = pd.Series(c); return ((s - s.rolling(n).mean()) / s.rolling(n).std(ddof=0)).to_numpy()

def families(D):
    c = D['c']; out = []
    for n, x, a30, rise, hold in itertools.product([4, 12, 24, 48, 96], [0.75, 1, 1.5, 2, 3], [0, 18, 25], [0, 3, 6, 12], [12, 24]):
        out.append(('MOM', dict(n=n, x=x, a30=a30, rise=rise, hold=hold), lambda n=n, x=x, a30=a30, rise=rise, hold=hold: g_mom(c, D['adx5'], D['adx30'], n, x, a30, rise, hold)))
    for n in [24, 48, 96, 144, 288, 576]:
        hh = pd.Series(D['h']).rolling(n).max().shift(1).to_numpy(); ll = pd.Series(D['l']).rolling(n).min().shift(1).to_numpy()
        for a30, hold in itertools.product([0, 20], [0, 12, 48]):
            out.append(('DONCHIAN', dict(n=n, a30=a30, hold=hold), lambda hh=hh, ll=ll, a30=a30, hold=hold: g_don(c, hh, ll, D['adx30'], a30, hold)))
    for ln in [7, 14, 21]:
        r = rsi(c, ln)
        for lo, ex, sl, mb in itertools.product([5, 10, 15, 20, 25], [40, 50, 60], [3, 5, 8], [72, 144, 288]):
            out.append(('RSI-MR', dict(len=ln, lo=lo, ex=ex, sl=sl, mb=mb), lambda r=r, lo=lo, ex=ex, sl=sl, mb=mb: g_osc(c, r, lo, 100 - lo, ex, 100 - ex, sl, mb)))
    for n in [48, 96, 288, 576]:
        z = zsc(c, n)
        for k, ex, sl, mb in itertools.product([2, 2.5, 3, 3.5], [0, -0.5, 0.5], [3, 5, 8], [144, 288]):
            out.append(('Z-MR', dict(n=n, k=k, ex=ex, sl=sl, mb=mb), lambda z=z, k=k, ex=ex, sl=sl, mb=mb: g_osc(c, z, -k, k, ex, -ex, sl, mb)))
    for f, s in itertools.product([12, 24, 48, 96], [96, 192, 288, 576, 1152]):
        if f >= s: continue
        ef = pd.Series(c).ewm(span=f, adjust=False).mean().to_numpy(); es = pd.Series(c).ewm(span=s, adjust=False).mean().to_numpy()
        out.append(('EMA', dict(f=f, s=s), lambda ef=ef, es=es: np.sign(ef - es).astype(np.int64)))
    for n, x, hold, lg in itertools.product([3, 6, 12, 24], [0.5, 1, 1.5, 2, 3], [12, 24, 48, 96], [0, 1]):
        out.append(('BTC-LİDER', dict(n=n, x=x, hold=hold, lag=lg), lambda n=n, x=x, hold=hold, lg=lg: g_btc(c, D['xc'], n, x, hold, lg)))
    for adx in [15, 20, 25, 30, 35, 40]:
        out.append(('BTC-4H', dict(adx=adx), lambda adx=adx: run1(D, P(btcAdx=adx), [0,0,0,1,0,0,0,0,0])))
    for px, oi, ex, sl, mb in itertools.product([1.5, 2, 3, 4], [1, 1.5, 2, 3], [50, 55, 60], [5, 8], [48, 72, 144]):
        out.append(('LİKİDASYON', dict(px=px, oi=oi, ex=ex, sl=sl, mb=mb), lambda px=px, oi=oi, ex=ex, sl=sl, mb=mb: run1(D, P(liqPx=px, liqOi=oi, liqExit=ex, liqSL=sl, liqBars=mb), [1,0,0,0,0,0,0,0,0])))
    for s, mh, cmode in itertools.product([0.35, 0.5, 0.65, 0.8, 1.0], [12, 24, 48], [0, 2]):
        out.append(('ÇARPIŞMA', dict(s=s, hold=mh, mode=cmode), lambda s=s, mh=mh, cmode=cmode: run1(D, P(aMom=3*s, bMom=2.5*s, minHold=mh), [0,0,1,0,0,0,0,0,cmode])))
    return out

def run1(D, Pv, f):
    return go(D, Pv, np.array(f, np.int64))['tgt']

if __name__ == '__main__':
    tf = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    D = cut(load(tf), SPLIT)  # IS only
    rows = []; t0 = time.time()
    for fam, prm, g in families(D):
        s = summ(simR(D, g())); s.update(fam=fam, prm=str(prm)); rows.append(s)
    df = pd.DataFrame(rows); df.to_pickle(f'lib_{tf}.pkl'); print(len(df), f'{time.time()-t0:.0f}s')
    pd.set_option('display.width', 200)
    for fam, g in df.groupby('fam', sort=False):
        ok = g[(g.n >= 100)]
        print(f'== {fam}: {len(g)} ayar | IS net>0: {(g.net>0).mean()*100:.0f}% | PF>1.15&n>=100: {((g.pf>1.15)&(g.n>=100)).sum()}')
        print(ok.sort_values('net', ascending=False).head(4)[['net', 'n', 'pf', 'mdd', 'prm']].to_string(index=False))
