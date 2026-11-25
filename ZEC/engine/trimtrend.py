# DEVAM EDEN TRENDDE: kârı FLAT ile al → yeniden gir; trend yokken TERS (V2.9). Hızlı katman testi.
import sys; sys.path.insert(0, '/home/claude/zecwork'); sys.argv = ['x', 'yok']
import pfsearch as P; import numpy as np, pandas as pd, numba, itertools
from pfsearch import o, h, l, c, posA, bEp, bXp, bXd, R10, tms, n, TICK, ix, S
mid = int(np.load('/home/claude/zecwork/base28.npz')['mid'])
Z4 = pd.DataFrame({'h': h, 'l': l, 'c': c}, ix).resample('4h').agg({'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
tr4 = pd.concat([Z4.h - Z4.l, (Z4.h - Z4.c.shift()).abs(), (Z4.l - Z4.c.shift()).abs()], axis=1).max(axis=1); A4 = tr4.ewm(alpha=1 / 14, adjust=False).mean().values
def map4(v):
    s_ = S(v, Z4.index + pd.Timedelta(hours=4)); j_ = np.searchsorted(s_.index.values, ix.values, side='right') - 1; out = s_.values[np.maximum(j_, 0)]; out[j_ < 0] = 0; return out
def state(up, E=200, Ne=55, Nx=20, tp=0.15, k=4):
    C = Z4.c.values; em = Z4.c.ewm(span=E, adjust=False).mean().values; e6 = np.r_[np.full(6, np.nan), em[:-6]]
    reg = (C > em) & (em > e6) if up else (C < em) & (em < e6)
    ext_n = Z4.h.rolling(Ne).max().shift(1).values if up else Z4.l.rolling(Ne).min().shift(1).values
    ex_x = Z4.l.rolling(Nx).min().shift(1).values if up else Z4.h.rolling(Nx).max().shift(1).values
    st_ = np.zeros(len(C)); s_ = 0; ep = ext = 0.0
    for t_ in range(len(C)):
        if s_ == 1:
            if up:
                ext = max(ext, C[t_]); bad = (not reg[t_]) or C[t_] < ex_x[t_] or (k > 0 and C[t_] < ext - k * A4[t_]) or (tp > 0 and C[t_] >= ep * (1 + tp))
            else:
                ext = min(ext, C[t_]); bad = (not reg[t_]) or C[t_] > ex_x[t_] or (k > 0 and C[t_] > ext + k * A4[t_]) or (tp > 0 and C[t_] <= ep * (1 - tp))
            if bad: s_ = 0
        elif reg[t_] and ((C[t_] > ext_n[t_]) if up else (C[t_] < ext_n[t_])): s_ = 1; ep = ext = C[t_]
        st_[t_] = s_
    return map4(st_) > 0
C4 = Z4.c.values; em4 = Z4.c.ewm(span=200, adjust=False).mean().values; e64 = np.r_[np.full(6, np.nan), em4[:-6]]
REGU = map4(((C4 > em4) & (em4 > e64)).astype(float)) > 0; REGD = map4(((C4 < em4) & (em4 < e64)).astype(float)) > 0
@numba.njit(cache=True)
def ovT(o, h, l, c, posA, bEp, bXp, bXd, R10, trU, trD, X, Y, Yt, tb, cont, contT, gain, flipOut, flipIn, tick):
    """Uçta kâr al: trend yönündeki pozisyonda (long & trU / short & trD) → FLAT (flipIn=0) ya da TERS; trend dışında flipOut.
    Trend içinde geri giriş eşiği Yt (long için RSI ≤ Yt) ve devam contT; gain>0 ise ayrıca pozisyon kârı ≥ gain% olunca da kâr al."""
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; rStp = 0.0; inT = False; fl = False; eq = np.zeros(n); nt = 0
    for i in range(1, n):
        if pos != 0:
            xp = -1.0
            if mode == 2 and fl and pos == -tSide:
                if (pos == 1 and l[i] <= rStp) or (pos == -1 and h[i] >= rStp): xp = (min(o[i], rStp) - tick) if pos == 1 else (max(o[i], rStp) + tick); mode = 0
            elif mode == 0 and not np.isnan(bXp[i]) and pos == bXd[i]: xp = bXp[i]
            if xp > 0: real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
        D = posA[i]; want = D
        if D != prevD: mode = 0
        elif D != 0:
            if mode == 0:
                tr = (D > 0 and trU[i]) or (D < 0 and trD[i])
                hit = (D > 0 and R10[i] >= X) or (D < 0 and R10[i] <= 100.0 - X)
                if gain > 0 and tr and pos == D and (c[i] / ep - 1.0) * D * 100.0 >= gain: hit = True
                if hit:
                    inT = tr; fl = flipIn if tr else flipOut
                    want = -D if fl else 0; mode = 2; tPx = c[i]; tBar = i; tSide = D; nt += 1
            else:
                yy = Yt if inT else Y; cc = contT if inT else cont
                bi = (R10[i] <= yy or i - tBar >= tb or c[i] >= tPx * (1.0 + cc)) if tSide > 0 else (R10[i] >= 100.0 - yy or i - tBar >= tb or c[i] <= tPx * (1.0 - cc))
                if bi: mode = 0
                else: want = (-tSide if pos == -tSide else 0) if fl else 0
        prevD = D
        if want != pos:
            if pos != 0: xp = c[i] - tick * pos; real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]
                if mode == 2 and fl and want == -tSide: rStp = ep * (1.0 - want * 0.02)
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq, nt
rows = []
def ad(lab, e, nt):
    x = e[::30]; d_ = np.maximum.accumulate(x) - x; ue = (S(e, ix).cummax() - S(e, ix)).to_numpy() >= 100
    t23 = np.searchsorted(tms, (pd.Timestamp('2023-02-06') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'))
    rows.append(dict(v=lab, kârAl=nt, net=e[-1], IS=e[mid], OOS=e[-1] - e[mid], yakın=e[-1] - e[t23], son300=e[-1] - e[np.searchsorted(tms, tms[-1] - 300 * 86400000)],
                     Ulcer=np.sqrt(((d_ / 5) ** 2).mean()), sualtı100=ue.mean() * 100, MaxDD=d_.max(), DD99=np.percentile(d_, 99)))
SU, SD = state(True), state(False)
TR = {'ara trend durumu': (SU, SD), 'EMA200 4s rejimi': (REGU, REGD)}
e, nt = ovT(o, h, l, c, posA, bEp, bXp, bXd, R10, SU, SD, 94.0, 70.0, 70.0, 240, 0.03, 0.03, 0.0, True, True, TICK); ad('V2.9: her zaman TERS', e, nt)
e, nt = ovT(o, h, l, c, posA, bEp, bXp, bXd, R10, SU, SD, 94.0, 70.0, 70.0, 240, 0.03, 0.03, 0.0, False, False, TICK); ad('her zaman FLAT (kâr al → yeniden gir)', e, nt)
for (tn, (tu, td)), Yt, contT, tb in itertools.product(TR.items(), (70.0, 80.0, 60.0), (0.01, 0.03), (60, 240)):
    e, nt = ovT(o, h, l, c, posA, bEp, bXp, bXd, R10, tu, td, 94.0, 70.0, Yt, tb, 0.03, contT, 0.0, True, False, TICK)
    ad(f'MELEZ ({tn}): trendde FLAT geri RSI≤{Yt:g} devam%{contT * 100:g} · dışında TERS · geri{tb}', e, nt)
for (tn, (tu, td)), g, Yt in itertools.product(TR.items(), (3.0, 5.0, 8.0), (60.0, 70.0)):
    e, nt = ovT(o, h, l, c, posA, bEp, bXp, bXd, R10, tu, td, 94.0, 70.0, Yt, 240, 0.03, 0.03, g, True, False, TICK)
    ad(f'MELEZ + trendde kâr ≥%{g:g} → FLAT, geri RSI≤{Yt:g} ({tn})', e, nt)
R = pd.DataFrame(rows); R.to_pickle('/home/claude/zecwork/trimtrend.pkl'); pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 82)
b = R.iloc[0]; R['dNet'] = R.net - b.net; R['dIS'] = R.IS - b.IS; R['dOOS'] = R.OOS - b.OOS
print(R.iloc[:2].round(1).to_string(index=False)); print()
print(R.iloc[2:].sort_values('dIS', ascending=False)[['v', 'kârAl', 'net', 'dNet', 'dIS', 'dOOS', 'yakın', 'son300', 'Ulcer', 'MaxDD', 'DD99']].round(1).to_string(index=False))
