import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, ztf, z01, z03
from path_quality import path
D = ztf.build(1); B = D['B']; c, h, l, o, a = D['c'], D['h'], D['l'], D['o'], D['atr']; n = D['n']; mid = B.index[D['mid']]
sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]
aL = (D['L'] >= 2); aS = (D['Sh'] >= 3) & ~aL; z = np.zeros(n, np.bool_)
r10 = (c / sh(c, 10) - 1) * 100
def go(spkX=None, B_on=False, bX=3.0, btrl=12.0, blkN=60):
    spk = np.nan_to_num(r10 >= spkX).astype(np.bool_) if spkX else z
    bL = (np.nan_to_num(r10 >= bX) & ~np.nan_to_num(sh(r10, 1) >= bX)).astype(np.bool_) if B_on else z
    eq, T = z03.sim7(o, h, l, c, a, aL.astype(np.bool_), aS.astype(np.bool_), 8.0 * D['r'], D['S'](480), bL, 8.0, float(btrl), 240, spk, int(blkN))
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own', 'guard']); t['et'] = B.index[t.ei.astype(int)]; t['xt'] = B.index[t.xi.astype(int)]
    return pd.Series(eq, index=B.index), t
def rep(lab, e, t):
    f = z01.metrics(e, t); A = z01.metrics(e, t, None, mid); Bm = z01.metrics(e, t, mid, None); sc = z01.sc4(e)
    y = t.groupby(t.et.dt.year).pnl.sum().round(0).astype(int).tolist()
    print(f"{lab}: net {f['net']:.0f} n {f['n']} PF {f['PF']:.2f} | MaxDD {f['MaxDD']:.0f} P99 {f['P99']:.0f} | Calmar {f['Calmar']:.2f} Sharpe {f['Sharpe']:.2f} SC4 {sc['SC4']:.3f} | σ10dk {f['s10']:.2f} σ1s {f['s1h']:.2f} | MAE maks %{f['maeMax']:.1f} | IS {A['net']:.0f} OOS {Bm['net']:.0f} | short {t[t.side<0].pnl.sum():.0f} (bekçi kapanış {int(t.guard.sum())}) | sıçrama long {t[t.own==2].pnl.sum():.0f} | yıllar {y}")
    return t
base = rep('R2-1dk BAZ', *go())
for X in (2.0, 3.0):
    rep(f'+ bekçi (10 bar ≥%{X:.0f} → short kapat, 60 bar short yok)', *go(spkX=X))
for bX, trl in itertools.product([2.0, 3.0], [8.0, 12.0]):
    rep(f'+ bekçi %2 + sıçrama long (≥%{bX:.0f}, iz {trl:.0f} ATR)', *go(spkX=2.0, B_on=True, bX=bX, btrl=trl))
e, t = go(spkX=2.0, B_on=True, bX=2.0, btrl=12.0)
w = t[(t.et >= '2026-11-22') | (t.xt >= '2026-11-22')]
print('\n23 KASIM OLAYI — işlemler (TR saati):'); print(w.assign(giriş=w.et + pd.Timedelta('3h'), çıkış=w.xt + pd.Timedelta('3h'), yön=np.where(w.side > 0, 'LONG', 'SHORT'),
      motor=np.where(w.own == 1, 'R2', 'SIÇRAMA'))[['giriş', 'çıkış', 'yön', 'motor', 'ep', 'xp', 'pnl', 'mae', 'guard']].round(3).to_string(index=False))
