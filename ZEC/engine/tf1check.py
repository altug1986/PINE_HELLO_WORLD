import sys; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, ztf, z01, z02
from zec_data import load
from path_quality import path
D = ztf.build(2); e2, t2 = ztf.run(D)
B1 = load(1, src='/home/claude/master/'); n1 = len(B1)
pos = B1.index.get_indexer(D['B'].index + pd.Timedelta('1min'))          # 2 dk barın kapanışını yapan 1 dk bar
ok = pos >= 0; tL = D['L'] >= 2; tS = (D['Sh'] >= 3) & ~tL
L1 = np.zeros(n1, np.bool_); S1 = np.zeros(n1, np.bool_); L1[pos[ok & tL]] = True; S1[pos[ok & tS]] = True
atr1 = np.full(n1, np.nan); atr1[pos[ok]] = D['atr'][ok]; atr1 = pd.Series(atr1).ffill().values
o, h, l, c = (B1[k].values.astype(float) for k in ('open', 'high', 'low', 'close')); z = np.zeros(n1, np.bool_)
eq, T = z02.sim4(o, h, l, c, atr1, z, z, np.full(n1, np.nan), L1, S1, 8.0, 960, 99.0)
t1 = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own']); t1['et'] = B1.index[t1.ei.astype(int)]
e1 = pd.Series(eq, index=B1.index); mid = D['B'].index[D['mid']]
for lab, e, t in (('2 dk sinyal · 2 dk bar içi', e2, t2), ('2 dk sinyal · 1 dk bar içi', e1, t1)):
    f = z01.metrics(e, t); A = z01.metrics(e, t, None, mid); Bm = z01.metrics(e, t, mid, None)
    print(f"{lab}: net {f['net']:.0f} n {f['n']} PF {f['PF']:.2f} | MaxDD {f['MaxDD']:.0f} P95/98/99 {f['P95']:.0f}/{f['P98']:.0f}/{f['P99']:.0f} | Calmar {f['Calmar']:.2f} | σ10dk {f['s10']:.2f} σ1s {f['s1h']:.2f} | MAE P90 %{f['maeP90']:.1f} maks %{f['maeMax']:.1f} | IS {A['net']:.0f} OOS {Bm['net']:.0f}")
# canlı sürüş 1 dk yolla (aynı girişler, 120 dk pencere, 2 ATR)
for lab, side in (('long', 1), ('short', -1)):
    ei = t1.ei[t1.side == side].astype(np.int64).values; ei = ei[ei < n1 - 200]
    P, rt, av, mae = path(h, l, c, atr1, ei, side, np.array([1.0, 2.0]), 120)
    rnd = np.sort(np.random.default_rng(5).integers(2000, n1 - 200, 4000)).astype(np.int64); P0, rt0, av0, _ = path(h, l, c, atr1, rnd, side, np.array([1.0, 2.0]), 120)
    print(f'1 dk yol, {lab} ({len(ei)}): P(önce lehe 2 ATR) %{np.nanmean(P[:,1])*100:.1f} (rastgele %{np.nanmean(P0[:,1])*100:.1f}) · doğru taraf %{np.nanmean(rt):.1f} ({np.nanmean(rt0):.1f}) · sürüş {np.nanmean(av):.1f} bps ({np.nanmean(av0):.1f}) · MAE P90 %{np.nanpercentile(mae,90):.1f}')
