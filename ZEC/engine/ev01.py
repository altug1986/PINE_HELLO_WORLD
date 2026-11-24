import sys; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01
B = load(2, src='/home/claude/master/'); F = z01.prep(B)
c = F['c']; a = F['atr']; n = len(c); mid = B.index[0] + (B.index[-1] - B.index[0]) / 2
isIS = (B.index < mid)
fw = {h: np.r_[c[h:] / c[:-h] - 1, np.full(h, np.nan)] * 1e4 for h in (15, 60, 240, 720)}
def rep(lab, sig, side):
    i = np.where(sig)[0]
    # aynı yönde 60 bar içinde tekrarları at (bağımsız olay)
    keep = []; last = -10**9
    for j in i:
        if j - last >= 60: keep.append(j); last = j
    i = np.array(keep, dtype=int)
    if len(i) < 30: print(f'{lab:34s} n={len(i)}'); return
    s = side[i] if hasattr(side, '__len__') else side
    out = [f'{lab:34s} n={len(i):5d}']
    for h in (15, 60, 240, 720):
        r = fw[h][i] * s
        out.append(f'h{h}: IS {np.nanmean(r[isIS[i]]):6.1f} OOS {np.nanmean(r[~isIS[i]]):6.1f}')
    print(' | '.join(out))
for N in (90, 360):
    HH, LL = z01.hl(F, N)
    up = (c > HH) & (np.r_[np.nan, c[:-1]] <= np.r_[np.nan, HH[:-1]]); dn = (c < LL) & (np.r_[np.nan, c[:-1]] >= np.r_[np.nan, LL[:-1]])
    rep(f'F1 Donchian{N} kırılım takip', up | dn, np.where(up, 1, -1))
for bars, k in ((10, 6), (10, 10), (30, 10), (30, 15)):
    r = (c - np.r_[np.full(bars, np.nan), c[:-bars]])
    z = r / a
    rep(f'F2 {bars}bar hareket ≥{k}ATR FADE', np.abs(z) >= k, -np.sign(z))
# F3 blow-off: son W bar içinde 20 dk getiri >= X% olmuş; tepe sonrası kapanış < son L bar dibi -> short (ayna long)
r10 = np.r_[np.full(10, np.nan), c[10:] / c[:-10] - 1] * 100
for X, W, L in ((5, 360, 15), (5, 360, 30), (8, 720, 30), (3, 180, 15)):
    spU = pd.Series(r10 >= X).rolling(W, min_periods=1).max().values > 0
    spD = pd.Series(r10 <= -X).rolling(W, min_periods=1).max().values > 0
    HHl, LLl = z01.hl(F, L)
    sS = spU & (c < LLl) & (np.r_[np.nan, c[:-1]] >= np.r_[np.nan, LLl[:-1]])
    sL = spD & (c > HHl) & (np.r_[np.nan, c[:-1]] <= np.r_[np.nan, HHl[:-1]])
    rep(f'F3 blowoff X{X}% W{W} L{L} ters', sS | sL, np.where(sS, -1, 1))
    rep(f'F3b spike sonrası DEVAM (ayna)', sS | sL, np.where(sS, 1, -1)) if X == 5 and L == 15 else None
for N, k in ((240, 8), (720, 12)):
    ema = pd.Series(c).ewm(span=N, adjust=False).mean().values; z = (c - ema) / a
    rep(f'F4 EMA{N} sapma ≥{k}ATR FADE', np.abs(z) >= k, -np.sign(z))
print('maliyet ≈ 8,3 bps gidiş-dönüş')
