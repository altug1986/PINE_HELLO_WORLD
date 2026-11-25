import sys, itertools, time; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, ztf, z01, z02
rows = []; t0 = time.time()
for thr in (15.0, 25.0, 40.0):
    D = ztf.build(1, thr=thr); n = D['n']; mid = D['mid']; z = np.zeros(n, np.bool_)
    for kL, kS, ts, tH in itertools.product([2, 3], [2, 3, 99], [6.0, 11.0, 16.0], [240, 480, 960]):
        tL = D['L'] >= kL; tS = (D['Sh'] >= kS) & ~tL
        eq, T = z02.sim4(D['o'], D['h'], D['l'], D['c'], D['atr'], z, z, np.full(n, np.nan), tL.astype(np.bool_), tS.astype(np.bool_), ts, tH, 99.0)
        t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own'])
        if len(t) < 300: continue
        rd = ztf.ride(D, t)
        x = eq[::30]; dd = (np.maximum.accumulate(x) - x).max(); xi = eq[:mid:30]; ddi = (np.maximum.accumulate(xi) - xi).max()
        y = t.groupby(D['B'].index.year.values[t.ei.astype(int)]).pnl.sum()
        rows.append(dict(thr=thr, kL=kL, kS=kS, ts=ts, tH=tH, n=len(t), sürIS=rd['IS'][2], sürOOS=rd['OOS'][2], doğIS=rd['IS'][1], doğOOS=rd['OOS'][1], P2IS=rd['IS'][0], P2OOS=rd['OOS'][0],
                         IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], net=eq[-1], MaxDD=dd, ISc=(eq[mid] / (mid / 525600)) / ddi if ddi > 0 else 0, Cal=(eq[-1] / (n / 525600)) / dd, maeMax=t.mae.max() * 100, posY=int((y > 0).sum())))
    print('thr', thr, 'bitti', round(time.time() - t0), 's', flush=True)
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/OPT_1DK.csv', index=False)
pd.set_option('display.width', 250)
# SEÇİM: önce canlı sürüş (IS), sonra kapanış (IS Calmar) — OOS sonra bakılır
R['sırala'] = R.sürIS.rank(ascending=False) + R.doğIS.rank(ascending=False) + R.ISc.rank(ascending=False)
print(R.sort_values('sırala').head(12).round(2).to_string(index=False))
print('\nOOS sürüş>0 & doğru>0 & OOS>0 oranı (tüm ayarlar): %', round(((R.sürOOS > 0) & (R.doğOOS > 0) & (R.OOS > 0)).mean() * 100))
