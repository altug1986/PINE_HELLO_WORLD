import numpy as np, pandas as pd
from feat import *
from v54 import target, S0, RA0, TICK
from met import SPLIT
P = dict(L=5.0, a=40.0, vk=2.0, cS=5.0, N=288)
tg = target(1.0, 1.5, 4.0)
# enstrümanlı katman: çıkış nedeni (1 konsorsiyum, 2 zarar sınırı %7, 3 kritik dönüş, 4 karşı poz bitişi)
sh = 0; ep = 0.; lock = 0; lext = 0.; prevT = 0; cnt = 0; cPx = 0.; cBar = 0; why = np.zeros(n, np.int8); out = np.zeros(n, np.int64)
for i in range(n):
    t_ = tg[i]; r = 1
    if t_ != prevT: lock = 0; cnt = 0; prevT = t_
    want = t_
    if cnt != 0:
        g = (c[i] / cPx - 1) * cnt * 100
        if g <= -P['cS'] or i - cBar >= P['N']: cnt = 0; r = 4
        else: want = cnt
    if cnt == 0:
        if sh != 0 and want == sh:
            loss = (c[i] / ep - 1) * sh * 100
            if loss <= -P['L'] and BADX[i] > P['a'] and BDI[i] * sh < 0 and NR[i] >= P['vk']: cnt = -sh; cPx = c[i]; cBar = i; want = cnt; r = 3
        if cnt == 0:
            if sh != 0 and want == sh and (c[i] / ep - 1) * sh * 100 <= -S0: lock = sh; lext = c[i]; want = 0; r = 2
            if lock != 0 and want == lock:
                lext = min(lext, c[i]) if lock > 0 else max(lext, c[i])
                if (c[i] / lext - 1) * lock * 100 >= RA0: lock = 0
                else: want = 0
    if want != sh:
        if sh != 0: why[i] = r
        sh = want; ep = c[i] + TICK * want
    out[i] = sh
from lib import simR
R = simR(D, out); X = pd.DataFrame(dict(eb=R['eb'], xb=R['xb'], dr=R['dir'], pnl=R['pnl'])); X['why'] = why[X.xb]; X['bars'] = X.xb - X.eb
X['x'] = pd.to_datetime(t[X.xb], unit='ms'); X['IS'] = t[X.xb] < SPLIT
print('motor net', round(X.pnl.sum()), len(X))
print('# çıkış nedeni:'); print(X.groupby('why').pnl.agg(['count', 'sum', 'mean']).round(1).to_string())
X['blk'] = (X.x.dt.year * 12 + X.x.dt.month - 1) // 2
lab = lambda b: f"{b*2//12}-{b*2%12+1:02d}"
for c_ in ['d288', 'adx15', 'nr', 'volr', 'm864', 'di15']: X[c_] = F[c_][X.eb]
X['d288A'] = X.d288 * X.dr; X['m864A'] = X.m864 * X.dr; X['di15A'] = X.di15 * X.dr
X['long'] = X.bars > 319
for b in [(2026*12+4)//2, (2022*12+10)//2, (2022*12+0)//2, (2026*12+2)//2, (2026*12+8)//2, (2020*12+8)//2]:
    s = X[X.blk == b]
    print(f"\n## {lab(b)} net {s.pnl.sum():.0f} | {len(s)} işlem | long {s.pnl[s.dr>0].sum():.0f} short {s.pnl[s.dr<0].sum():.0f} | >319 bar: {s.long.sum()} işlem {s.pnl[s.long].sum():.0f}$ | çıkış nedeni:",
          s.groupby('why').pnl.sum().round(0).to_dict(), '| volr ort', round(s.volr.mean(), 2), 'nr ort', round(s.nr.mean(), 2), 'adx15 ort', round(s.adx15.mean(), 1))
    print(s.nsmallest(5, 'pnl')[['x', 'dr', 'pnl', 'bars', 'why', 'd288A', 'm864A', 'di15A', 'adx15', 'nr', 'volr']].round(2).to_string(index=False))
# saat ve gün IS/OOS
X['hr'] = pd.to_datetime(t[X.eb], unit='ms').hour; X['wd'] = pd.to_datetime(t[X.eb], unit='ms').weekday
h = X.groupby(['hr', 'IS']).pnl.sum().unstack().round(0); print('\n# saat IS/OOS:', ' '.join(f"{i}:{r[True]:.0f}/{r[False]:.0f}" for i, r in h.iterrows()))
w = X.groupby(['wd', 'IS']).pnl.sum().unstack().round(0); print('# gün IS/OOS:', ' '.join(f"{i}:{r[True]:.0f}/{r[False]:.0f}" for i, r in w.iterrows()))
X.to_pickle('diag2.pkl')
