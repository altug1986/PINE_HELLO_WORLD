# Sualtı anatomisi + simetrik trend koruması (4 s düşüş trendinde long yok)
import sys; sys.path.insert(0, '/home/claude/zecwork'); sys.argv = ['x', 'yok']
import pfsearch as P; import numpy as np, pandas as pd, numba, itertools
from pfsearch import o, h, l, c, posA, bEp, bXp, bXd, R10, tms, n, TICK, ix, S
Z = np.load('/home/claude/zecwork/base28.npz'); TB = Z['T']; mid = int(Z['mid'])
e9 = P.ov5(o, h, l, c, posA, bEp, bXp, bXd, R10, P.F0, 0, 0, 0.0, P.ZR60 < 0, P.F0, 0, TICK)
s = S(e9, ix)[::30]; dd = (s.cummax() - s).values; t = s.index; eps = []; i = 0
while i < len(dd):
    if dd[i] > 150:
        j = i
        while j < len(dd) and dd[j] > 0: j += 1
        k = i + int(np.argmax(dd[i:j])); st = k
        while st > 0 and dd[st - 1] > 0: st -= 1
        eps.append((t[st], t[k], dd[k])); i = j
    else: i += 1
own = {0: 'satır0', 1: 'E0', 2: 'E1', 3: 'E2', 4: 'E3', 5: 'POMPA'}
print('Sualtı bölümleri (V2.9 > 150 $):')
for a, b, d in sorted(eps, key=lambda x: -x[2])[:10]:
    ia, ib = np.searchsorted(tms, a.value // 10 ** 6), np.searchsorted(tms, b.value // 10 ** 6)
    m = (TB[:, 0] >= ia) & (TB[:, 0] < ib); x = TB[m]
    lp = x[x[:, 2] > 0, 5].sum(); sp = x[x[:, 2] < 0, 5].sum(); byo = {own[int(k)]: round(x[x[:, 7] == k, 5].sum()) for k in np.unique(x[:, 7])}
    print(f'  {a:%Y-%m-%d} → {b:%Y-%m-%d} derinlik {d:.0f} $ · ZEC {(c[ib] / c[ia] - 1) * 100:+.1f}% · baz long {lp:+.0f} / short {sp:+.0f} · motor {byo}')
# 4 s düşüş trendi durumu (V2.8'deki yükseliş trendinin aynası)
Z4 = pd.DataFrame({'h': h, 'l': l, 'c': c}, ix).resample('4h').agg({'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
tr4 = pd.concat([Z4.h - Z4.l, (Z4.h - Z4.c.shift()).abs(), (Z4.l - Z4.c.shift()).abs()], axis=1).max(axis=1); A4 = tr4.ewm(alpha=1 / 14, adjust=False).mean().values
def map4(v):
    s_ = S(v, Z4.index + pd.Timedelta(hours=4)); j_ = np.searchsorted(s_.index.values, ix.values, side='right') - 1; out = s_.values[np.maximum(j_, 0)]; out[j_ < 0] = 0; return out
def dstate(E, Ne, Nx, tp, k):
    C = Z4.c.values; em = Z4.c.ewm(span=E, adjust=False).mean().values; reg = (C < em) & (em < np.r_[np.full(6, np.nan), em[:-6]])
    lo = Z4.l.rolling(Ne).min().shift(1).values; hi = Z4.h.rolling(Nx).max().shift(1).values; st_ = np.zeros(len(C)); s_ = 0; ep = ext = 0.0
    for t_ in range(len(C)):
        if s_ == 1:
            ext = min(ext, C[t_])
            if (not reg[t_]) or C[t_] > hi[t_] or (k > 0 and C[t_] > ext + k * A4[t_]) or (tp > 0 and C[t_] <= ep * (1 - tp)): s_ = 0
        elif reg[t_] and C[t_] < lo[t_]: s_ = 1; ep = ext = C[t_]
        st_[t_] = s_
    return map4(st_) > 0
@numba.njit(cache=True)
def ov8(o, h, l, c, posA, bEp, bXp, bXd, R10, vL, tick):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; rStp = 0.0; eq = np.zeros(n)
    for i in range(1, n):
        if pos != 0:
            xp = -1.0
            if mode == 2 and pos == -tSide:
                if (pos == 1 and l[i] <= rStp) or (pos == -1 and h[i] >= rStp): xp = (min(o[i], rStp) - tick) if pos == 1 else (max(o[i], rStp) + tick); mode = 0
            elif not np.isnan(bXp[i]) and pos == bXd[i]: xp = bXp[i]
            if xp > 0: real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
        D = posA[i]; want = D
        if D != prevD: mode = 0
        elif D != 0:
            if mode == 0:
                if (D > 0 and R10[i] >= 94) or (D < 0 and R10[i] <= 6): want = -D; mode = 2; tPx = c[i]; tBar = i; tSide = D
            else:
                bi = (R10[i] <= 70 or i - tBar >= 240 or c[i] >= tPx * 1.03) if tSide > 0 else (R10[i] >= 30 or i - tBar >= 240 or c[i] <= tPx * 0.97)
                if bi: mode = 0
                else: want = -tSide if pos == -tSide else 0
        prevD = D
        if vL[i] and want == 1: want = 0
        if want != pos:
            if pos != 0: xp = c[i] - tick * pos; real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]
                if mode == 2 and want == -tSide: rStp = ep * (1.0 - want * 0.02)
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq
def ulc(e): x = e[::30]; d = (np.maximum.accumulate(x) - x) / 5; return np.sqrt((d ** 2).mean())
rows = []
def ad(lab, e):
    x = e[::30]; d_ = np.maximum.accumulate(x) - x; ue = (S(e, ix).cummax() - S(e, ix)).to_numpy() >= 100
    rows.append(dict(v=lab, net=e[-1], IS=e[mid], OOS=e[-1] - e[mid], Ulcer=ulc(e), sualtı100=ue.mean() * 100, MaxDD=d_.max(), DD99=np.percentile(d_, 99), son300=e[-1] - e[np.searchsorted(tms, tms[-1] - 300 * 86400000)]))
ad('V2.9', e9)
for E, (Ne, Nx), tp, k in itertools.product((100, 200), ((20, 10), (55, 20)), (0.10, 0.15, 0.25, 0.0), (4, 6)):
    vL = dstate(E, Ne, Nx, tp, k); ad(f'düşüş trendi EMA{E} kırılım{Ne}/{Nx} kâr%{int(tp * 100)} iz{k} → long yok (gerçek emir katmanı)', ov8(o, h, l, c, posA, bEp, bXp, bXd, R10, vL, TICK))
R = pd.DataFrame(rows); pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 80); R.to_pickle('/home/claude/zecwork/ulcer2.pkl')
print(R.sort_values('IS', ascending=False).round(2).head(16).to_string(index=False))
