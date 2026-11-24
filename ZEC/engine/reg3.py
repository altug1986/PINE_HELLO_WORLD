import sys, json, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02, regime
B = load(2, src='/home/claude/master/'); F = z01.prep(B); c = F['c']; n = len(c)
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json')); ix = {k: j for j, k in enumerate(names)}
midT = B.index[0] + (B.index[-1] - B.index[0]) / 2; mid = np.searchsorted(B.index.values, midT.to_datetime64())
A4 = np.load('/home/claude/zecwork/adx_4h.npy'); adx, di = A4[:, 0], A4[:, 1]
f240 = np.r_[c[240:] / c[:-240] - 1, np.full(240, np.nan)] * 1e4
tp = np.log(B.tt_pos_ls.values); d4 = tp - np.r_[np.full(120, np.nan), tp[:-120]]; d1g = tp - np.r_[np.full(720, np.nan), tp[:-720]]
qv = pd.Series(B.quote_volume.values).rolling(30).sum().values; wn = pd.Series(B.wh_buy_q50k.values - B.wh_sell_q50k.values).rolling(30).sum().values / qv
S = {'ttposK4h': -np.sign(np.nan_to_num(d4)) * (np.abs(np.nan_to_num(d4)) > 0.03), 'ttposK1g': -np.sign(np.nan_to_num(d1g)) * (np.abs(np.nan_to_num(d1g)) > 0.06),
     'whaleT': np.sign(np.nan_to_num(wn)) * (np.abs(np.nan_to_num(wn)) > 0.05)}
for k in ['X17_basis', 'P09_spike', 'R21_glb', 'P07_rsi', 'P08_zema', 'P10_bb', 'P12_oiLiq', 'P11_oiCont', 'P15_resid', 'R18_whale', 'R19_book', 'P04_htf4h']:
    S[k] = M[:, ix[k]].astype(float)
def build(Ath, thr):
    R = np.where(adx >= Ath, di, np.where(adx < 20, 0, 9))
    L = np.zeros(n); Sh = np.zeros(n); chosen = []
    for k, v in S.items():
        for side in (1, -1):
            for lab, val in (('UP', 1), ('DN', -1), ('YAT', 0)):
                m = (v == side) & (R == val)
                if m[:mid].sum() < 200: continue
                ic = np.nanmean((f240 * side)[:mid][m[:mid]])          # SEÇİM YALNIZ IS
                if ic >= thr:
                    chosen.append(f'{k}{"L" if side > 0 else "S"}@{lab}')
                    (L if side > 0 else Sh)[m] += 1
    return L, Sh, chosen, R
def atr4(H, k=14):
    c1 = H.close.shift(1); tr = np.maximum(H.high - H.low, np.maximum((H.high - c1).abs(), (H.low - c1).abs())); return tr.ewm(alpha=1 / k, adjust=False).mean().values
AT4 = z02.htf(B, '4h', atr4); hh = z02.htf(B, '4h', lambda H: H.high.rolling(10).max().values)
z = np.zeros(n, np.bool_)
def fast(e, lo, hi):
    x = e[lo:hi] - e[lo]; x30 = x[::15]; dd = np.maximum.accumulate(x30) - x30
    return x[-1], (x[-1] / ((hi - lo) / (720 * 365))) / dd.max() if dd.max() > 0 else 0, dd.max()
rows = []
for Ath, thr in itertools.product([20, 25], [10, 25]):
    L, Sh, chosen, R = build(Ath, thr)
    print(f'ADX≥{Ath} IS eşik {thr} bps → {len(chosen)} kural:', ', '.join(chosen))
    for core in ('yok', 'Donchian', 'Donchian@UP'):
        eL = z if core == 'yok' else (np.nan_to_num(c > hh).astype(np.bool_) & ((R == 1) if core == 'Donchian@UP' else True))
        for kL, kS, ts, tH in itertools.product([1, 2, 3], [1, 2, 99], [4, 8], [120, 240, 480]):
            tL = (L >= kL); tS = (Sh >= kS) & ~tL
            eq, T = z02.sim4(F['o'], F['h'], F['l'], c, F['atr'], eL.astype(np.bool_), z, AT4 * 4, tL.astype(np.bool_), tS.astype(np.bool_), float(ts), int(tH), 99.0)
            t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own'])
            a = fast(eq, 0, mid); b = fast(eq, mid, n); f = fast(eq, 0, n)
            yrs = pd.Series(t.pnl.values, index=B.index[t.ei.astype(int)]).groupby(lambda x: x.year).sum() if len(t) else pd.Series(dtype=float)
            rows.append(dict(A=Ath, thr=thr, core=core, kL=kL, kS=kS, ts=ts, tH=tH, n=len(t), net=eq[-1], IS=a[0], OOS=b[0], MaxDD=f[2], Cal=f[1], ISc=a[1],
                             L=t[t.side > 0].pnl.sum(), Sh=t[t.side < 0].pnl.sum(), posY=int((yrs > 0).sum()), minY=yrs.min() if len(yrs) else 0))
R_ = pd.DataFrame(rows); R_.to_csv('/home/claude/zecwork/reg3.csv', index=False)
print(R_.sort_values('ISc', ascending=False).head(12).round(1).to_string(index=False))
print('\nIS>0 & OOS>0 & 6/6 yıl pozitif:'); print(R_[(R_.IS > 0) & (R_.OOS > 0) & (R_.posY == 6)].sort_values('Cal', ascending=False).head(12).round(1).to_string(index=False))
