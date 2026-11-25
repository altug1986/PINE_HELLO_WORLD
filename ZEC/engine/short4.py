import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
src = open('/home/claude/pine_hello_world/ZEC/engine/short_scan.py').read().split("rows = []; base = {}")[0]; exec(src)
from path_quality import path
DN = (RG == -1)
sets = {'DN çekirdek (pump≥10 · RSI>85 · basis z≥2 · OI15 yeni short)': (T['A3 Pump fade 20bar≥10ATR'] | T['A5 RSI>85 çıkış'] | T['C1 Basis primi z≥2'] | T['B1 Yeni short parası OI15 z≥1.5']) & DN,
        'DN seçkin (pump≥10 · RSI>85)': (T['A3 Pump fade 20bar≥10ATR'] | T['A5 RSI>85 çıkış']) & DN,
        'Basis z≥2 (tüm rejim)': T['C1 Basis primi z≥2']}
z = np.zeros(n, np.bool_); zf = np.zeros(n, np.bool_)
for lab, sig in sets.items():
    e = dedupe(sig); e = e[(e > 2000) & (e < n - 800)]
    P, rt, av, mae = path(h, l, c, a, e.astype(np.int64), -1, np.array([1.0, 2.0, 3.0]), 60)
    isk = e < mid
    print(f'\n== {lab}: olay {len(e)} | P(önce lehe 1/2/3 ATR) IS {np.nanmean(P[isk],0).round(1)} OOS {np.nanmean(P[~isk],0).round(1)} (rastgele ≈50) | doğru taraf %{np.nanmean(rt[isk]):.1f}/{np.nanmean(rt[~isk]):.1f} | sürüş bps {np.nanmean(av[isk]):.1f}/{np.nanmean(av[~isk]):.1f}')
    for k, H in itertools.product([2, 3, 4], [60, 240]):
        eq, T_ = z02.sim6(F['o'], h, l, c, a, z, np.nan_to_num(sig).astype(np.bool_), float(k), int(H), 0.0, 0.0, float(k), 0.0, zf, np.full(n, np.nan), zf, zf)
        t = pd.DataFrame(T_, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'mfe', 'tr']); ik = t.ei < mid; x30 = eq[::15]
        yy = t.groupby(yr[t.ei.astype(int)]).pnl.sum().reindex(range(2021, 2027)).fillna(0)
        print(f'   simetrik çıkış ±{k} ATR, süre {H:3d}: n {len(t):5d} net {eq[-1]:6.0f} (IS {t.pnl[ik].sum():5.0f} / OOS {t.pnl[~ik].sum():5.0f}) kazanma %{(t.pnl>0).mean()*100:.0f} MaxDD {(np.maximum.accumulate(x30)-x30).max():4.0f} yıllar {yy.round(0).astype(int).tolist()}')
