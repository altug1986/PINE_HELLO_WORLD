# E0 (ve tüm motorlar) için zaman-zarar kesici: baz işlem B bar sonra −x% altındaysa gerçek pozisyon kapanır, baz işlem bitene kadar flat
import sys; sys.path.insert(0, '/home/claude/zecwork'); sys.argv = ['x', 'yok']
import pfsearch as P; import numpy as np, pandas as pd, numba, itertools
from pfsearch import o, h, l, c, posA, bEp, bXp, bXd, R10, tms, n, TICK, ix, S
Z = np.load('/home/claude/zecwork/base28.npz'); TB = Z['T']; mid = int(Z['mid'])
ownA = np.full(n, -1, np.int8); bEi = np.zeros(n, np.int64)
for r in TB:
    ei, xi = int(r[0]), int(r[1]); ownA[ei:xi] = int(r[7]); bEi[ei:xi] = ei
@numba.njit(cache=True)
def ov9(o, h, l, c, posA, bEp, bXp, bXd, R10, ownA, bEi, tgt, B, x, tick):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; rStp = 0.0; eq = np.zeros(n); cut = -1; ncut = 0
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
        if D != 0 and (tgt < 0 or ownA[i] == tgt) and i - bEi[i] >= B and (c[i] / bEp[i] - 1.0) * D * 100.0 <= -x and cut != bEi[i]:
            cut = bEi[i]; ncut += 1
        if D != 0 and cut == bEi[i]: want = 0
        if want != pos:
            if pos != 0: xp = c[i] - tick * pos; real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]
                if mode == 2 and want == -tSide: rStp = ep * (1.0 - want * 0.02)
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq, ncut
def ulc(e): x = e[::30]; d = (np.maximum.accumulate(x) - x) / 5; return np.sqrt((d ** 2).mean())
rows = []
def ad(lab, e, nc):
    x = e[::30]; d_ = np.maximum.accumulate(x) - x; ue = (S(e, ix).cummax() - S(e, ix)).to_numpy() >= 100
    rows.append(dict(v=lab, kesim=nc, net=e[-1], IS=e[mid], OOS=e[-1] - e[mid], Ulcer=ulc(e), sualtı100=ue.mean() * 100, MaxDD=d_.max(), DD99=np.percentile(d_, 99), son300=e[-1] - e[np.searchsorted(tms, tms[-1] - 300 * 86400000)]))
e, nc = ov9(o, h, l, c, posA, bEp, bXp, bXd, R10, ownA, bEi, 99, 10 ** 9, 99.0, TICK); ad('V2.9', e, nc)
for tg, B, x in itertools.product((1, -1), (120, 240, 480, 960, 1440), (1.0, 2.0, 3.0, 5.0)):
    e, nc = ov9(o, h, l, c, posA, bEp, bXp, bXd, R10, ownA, bEi, tg, B, x, TICK); ad(f'{"E0" if tg == 1 else "tüm motorlar"}: {B} bar sonra ≤ −%{x:g} ise kes', e, nc)
R = pd.DataFrame(rows); R.to_pickle('/home/claude/zecwork/ulcer3.pkl'); pd.set_option('display.width', 250)
print(R.iloc[:1].round(2).to_string(index=False)); print(R.iloc[1:].sort_values('IS', ascending=False).head(14).round(2).to_string(index=False))
print('IS artı:', (R.IS.iloc[1:] > R.IS.iloc[0]).sum(), '/', len(R) - 1, '· iki yarı artı:', ((R.IS.iloc[1:] > R.IS.iloc[0]) & (R.OOS.iloc[1:] > R.OOS.iloc[0])).sum(), '· Ulcer düşen:', (R.Ulcer.iloc[1:] < R.Ulcer.iloc[0]).sum())
