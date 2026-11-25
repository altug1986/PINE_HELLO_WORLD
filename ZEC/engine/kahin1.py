import sys, time, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, kahin, z01, z02
from zec_data import load
t0 = time.time(); B = load(1, src='/home/claude/master/'); Xf, atr = kahin.features(B); import gc; gc.collect(); print('özellik', Xf.shape, round(time.time() - t0), 's', flush=True)
p, y = kahin.walk_forward(B, Xf, H=120); print('yürüyen eğitim bitti', round(time.time() - t0), 's', flush=True)
np.save('/home/claude/zecwork/kahin_p120.npy', p)
ok = ~np.isnan(p) & ~np.isnan(y); yr = B.index.year.values
q = pd.qcut(p[ok], 10, labels=False); fy = y[ok] * 1e4
cal = pd.DataFrame({'q': q, 'p': p[ok], 'f': fy, 'y': yr[ok]}).groupby('q').agg(p=('p', 'mean'), ileri_bps=('f', 'mean'), yukarı=('f', lambda s: (s > 0).mean() * 100))
print('\nKAHİN TESTİ (tamamı örneklem dışı, 2022+): olasılık dilimi → gerçekleşen 2 sa getiri'); print(cal.round(3).T.to_string())
for y_ in range(2022, 2027):
    m = ok & (yr == y_); qq = pd.qcut(p[m], 10, labels=False); f = y[m] * 1e4
    print(f'  {y_}: en düşük dilim {f[qq == 0].mean():6.1f} bps · en yüksek dilim {f[qq == 9].mean():6.1f} bps · fark {f[qq == 9].mean() - f[qq == 0].mean():6.1f}')
c, h, l, o = (B[k].values.astype(float) for k in ('close', 'high', 'low', 'open'))
S = np.nan_to_num((p - 0.5) * 100, nan=0.0)
rows = []
for tin, tout, side in itertools.product([0.55, 0.58, 0.62, 0.66, 0.70], [0.50, 0.53], ['çift', 'long']):
    if tout >= tin: continue
    SS = S if side == 'çift' else np.maximum(S, -(tin - 0.5) * 100 + 0.01)
    eq, T = z02.sim3(o, h, l, c, atr.astype(float), SS, (tin - 0.5) * 100, (tout - 0.5) * 100, 1e6, 10**9)
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae']); t['et'] = B.index[t.ei.astype(int)]
    e = pd.Series(eq, index=B.index).loc['2022-01-01':]; e = e - e.iloc[0]; tt = t[t.et >= '2022-01-01']
    if len(tt) < 50: continue
    m30 = e.resample('30min').last().dropna(); dd = (m30.cummax() - m30).max(); yrs = (e.index[-1] - e.index[0]).days / 365
    r2 = np.corrcoef(np.arange(len(m30)), m30.values)[0, 1] ** 2
    yy = tt.groupby(tt.et.dt.year).pnl.sum().round(0).astype(int).tolist()
    rows.append(dict(giriş=tin, çıkış=tout, yön=side, n=len(tt), net=e.iloc[-1], pct=e.iloc[-1] / 5, PF=tt.pnl[tt.pnl > 0].sum() / -tt.pnl[tt.pnl < 0].sum(), MaxDD=dd, Calmar=e.iloc[-1] / yrs / dd,
                     R2=r2, maeMax=tt.mae.max() * 100, ortTut=(tt.xi - tt.ei).mean() / 60, yıllar=yy, short=tt[tt.side < 0].pnl.sum()))
R = pd.DataFrame(rows); pd.set_option('display.width', 250)
print('\nTAHMİNLE GİR / TAHMİNLE ÇIK (SL yok, iz yok), 2022-01 → 2026-11, sabit 500 $:'); print(R.round(2).to_string(index=False))
R.to_csv('/home/pine_hello_world_tmp.csv', index=False) if False else R.to_csv('/home/claude/pine_hello_world/ZEC/research/KAHIN_H120.csv', index=False)
