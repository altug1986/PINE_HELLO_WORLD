import sys, itertools, json; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02
B = load(2, src='/home/claude/master/'); F = z01.prep(B); n = len(B); c = F['c']
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json')); ix = {k: j for j, k in enumerate(names)}
def atr4(H, k=14):
    c1 = H.close.shift(1); tr = np.maximum(H.high - H.low, np.maximum((H.high - c1).abs(), (H.low - c1).abs())); return tr.ewm(alpha=1 / k, adjust=False).mean().values
A4 = z02.htf(B, '4h', atr4)
hh = z02.htf(B, '4h', lambda H: H.high.rolling(10).max().values); ll = z02.htf(B, '4h', lambda H: H.low.rolling(10).min().values)
eL = np.nan_to_num(c > hh).astype(np.bool_); eS0 = np.nan_to_num(c < ll).astype(np.bool_); z = np.zeros(n, np.bool_)
w = np.zeros(len(names)); w[[ix[k] for k in ['P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P12_oiLiq', 'X17_basis', 'P04_htf4h']]] = 1
S = M.astype(float) @ w; vL = S >= 3; vS = S <= -3
fL, fS = z01.fade_sig(F, 20, 10)
def fast(e, lo, hi):
    x = e[lo:hi] - e[lo]; x30 = x[::15]; dd = np.maximum.accumulate(x30) - x30
    return x[-1], (x[-1] / ((hi - lo) / (720 * 365))) / dd.max() if dd.max() > 0 else 0, dd.max()
rows = []
for k, tr_side, tac, ts, tH, tD in itertools.product([4, 6, 8], ['long', 'çift'], ['yok', 'oyL', 'oyL+fade', 'oyLS+fade'], [4, 8, 12], [120, 480], [99.0, 6.0]):
    if tac == 'yok' and (ts, tH, tD) != (4, 120, 99.0): continue
    eS = eS0 if tr_side == 'çift' else z
    tL = {'yok': z, 'oyL': vL, 'oyL+fade': vL | fL, 'oyLS+fade': vL | fL}[tac]
    tS = {'yok': z, 'oyL': z, 'oyL+fade': z, 'oyLS+fade': vS | fS}[tac]
    tL = tL & ~tS; tS = tS & ~tL
    eq, T = z02.sim4(F['o'], F['h'], F['l'], c, F['atr'], eL, eS, A4 * k, tL, tS, float(ts), int(tH), float(tD))
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own'])
    a = fast(eq, 0, mid); b = fast(eq, mid, n); f = fast(eq, 0, n)
    rows.append(dict(trend=f'{k}xATR4h {tr_side}', tac=tac, ts=ts, tH=tH, tD=tD, n=len(t), net=eq[-1], pct=eq[-1] / 5, IS=a[0], OOS=b[0], MaxDD=f[2], Cal=f[1],
                     trend_pnl=t[t.own == 1].pnl.sum(), tac_pnl=t[t.own == 2].pnl.sum(), top3=t.pnl.nlargest(3).sum() / max(eq[-1], 1), maemax=t.mae.max() * 100))
R = pd.DataFrame(rows); R.to_csv('/home/claude/zecwork/combo1.csv', index=False)
print(R.sort_values('net', ascending=False).head(10).round(2).to_string(index=False))
print(R[(R.IS > 0) & (R.OOS > 0)].sort_values('Cal', ascending=False).head(10).round(2).to_string(index=False))
