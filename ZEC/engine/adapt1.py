"""UYARLANIR OY: her bileşen × rejim için ağırlık = son W günün gerçekleşmiş bilgi katsayısı (2 gün gecikmeli, geleceği görmez).
Piyasa davranışı değişince (2026 gibi) oylar kendini çevirir. Tarih filtresi yok."""
import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, z03
exec(open('/home/claude/zecwork/bob1.py').read().split("blk = np.searchsorted")[0])
day = ((idx - idx[0].normalize()) // pd.Timedelta('1D')).values.astype(int); nd = day.max() + 1
comps = []
for k in order:
    for val in (1, -1, 0):
        comps.append((k, val, (V[k] * (RG == val)).astype(np.float32)))
blk = np.searchsorted(idx.values, pd.date_range(idx[0].normalize(), idx[-1], freq='14D').values)
pos = lambda t_: np.searchsorted(idx.values, np.datetime64(t_))
WIN = {'son2ay': (pos('2026-09-24'), n - 1), 'son3hf': (pos('2026-11-03'), n - 1), '23Kas': (pos('2026-11-23'), n - 1), 'zayıf': (pos('2022-06-01'), pos('2023-04-01'))}
rows = []
for hz in (240, 480):
    f = np.nan_to_num(F[hz]); fin = ~np.isnan(F[hz])
    DS = np.zeros((nd, len(comps))); DC = np.zeros((nd, len(comps)))
    for j, (_, _, vv) in enumerate(comps):
        DS[:, j] = np.bincount(day, weights=f * vv * fin, minlength=nd); DC[:, j] = np.bincount(day, weights=(vv != 0) * fin, minlength=nd)
    for W, thr in itertools.product([14, 30, 60, 120], [0.0, 5.0, 15.0]):
        cs = np.cumsum(DS, 0); cc = np.cumsum(DC, 0)
        rs = cs - np.r_[np.zeros((W, len(comps))), cs[:-W]]; rc = cc - np.r_[np.zeros((W, len(comps))), cc[:-W]]
        ic = np.where(rc > 200, rs / np.maximum(rc, 1), 0.0); ic = np.r_[np.zeros((2, len(comps))), ic[:-2]]      # 2 gün gecikme
        w = np.where(np.abs(ic) >= thr, np.sign(ic), 0.0)
        S = np.zeros(n, np.float32)
        for j, (_, _, vv) in enumerate(comps):
            S += (w[day, j] * vv).astype(np.float32)
        for kk, ts, tH in itertools.product([1, 2, 3], [11.0, 16.0], [480, 960, 1920]):
            aL = S >= kk; aS = S <= -kk
            eq, T = z03.sim7(o, h, l, c, a, aL, aS, ts, tH, z, 8.0, 12.0, 240, spk, 60)
            if len(T) < 300: continue
            bp = np.diff(eq[blk]); x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30
            r = dict(hz=hz, W=W, thr=thr, k=kk, ts=ts, tH=tH, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(),
                     R2=np.corrcoef(np.arange(len(x30)), x30)[0, 1] ** 2 if eq[-1] > 0 else 0, blokMed=np.median(bp), blokPoz=(bp > 0).mean() * 100,
                     maeP95=np.percentile(T[:, 6], 95) * 100, short=T[T[:, 2] < 0, 5].sum())
            for nm, (a_, b_) in WIN.items():
                r[nm] = eq[b_] - eq[a_]
            rows.append(r)
    print('hz', hz, 'bitti', len(rows), flush=True)
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/ADAPT_1DK.csv', index=False); pd.set_option('display.width', 260)
print('R5 referans: net 4.249 · son2ay −213 · son3hf −128 · 23Kas +1 · zayıf +166')
print('\nEn yüksek net:'); print(R.sort_values('net', ascending=False).head(8).round(1).to_string(index=False))
G = R[(R.IS > 0) & (R.OOS > 0) & (R.son2ay > 0) & (R.zayıf > 0)]
print(f'\nİki yarı + son 2 ay + zayıf dönem pozitif ({len(G)}):'); print(G.sort_values('net', ascending=False).head(10).round(1).to_string(index=False))
