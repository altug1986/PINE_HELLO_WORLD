import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, z03, ztf
exec(open('/home/claude/zecwork/bob1.py').read().split("blk = np.searchsorted")[0])
import importlib; importlib.reload(z03)
B4 = B[['open', 'high', 'low', 'close']]
def atr4(H, k=14):
    c1 = H.close.shift(1); tr = np.maximum(H.high - H.low, np.maximum((H.high - c1).abs(), (H.low - c1).abs())); return tr.ewm(alpha=1 / k, adjust=False).mean().values
A4 = ztf.htf(B4, 1, '4h', atr4)
blk = np.searchsorted(idx.values, pd.date_range(idx[0].normalize(), idx[-1], freq='14D').values)
pos_ = lambda t_: np.searchsorted(idx.values, np.datetime64(t_))
WIN = {'son2ay': (pos_('2026-09-24'), n - 1), 'son3hf': (pos_('2026-11-03'), n - 1), '23Kas': (pos_('2026-11-23'), n - 1), 'zayıf': (pos_('2022-06-01'), pos_('2023-04-01'))}
RGi = np.nan_to_num(RG).astype(np.int64); rows = []
for (thr, hz, kL, kS), mode, k4 in itertools.product([(15, 240, 2, 3), (10, 480, 2, 3), (10, 480, 2, 4), (25, 480, 2, 3), (15, 240, 2, 4), (10, 120, 2, 2)], [0, 1, 2], [3, 4, 6, 8]):
    if mode == 0 and k4 != 3: continue
    L, S = select(thr, hz); aL = (L >= kL); aS = (S >= kS) & ~aL
    for ts, tH in ((16.0, 1920), (11.0, 960), (24.0, 1920)):
        eq, T = z03.sim10(o, h, l, c, a, aL, aS, ts, tH, spk, 60, RGi, A4 * k4, mode)
        bp = np.diff(eq[blk]); x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; tp = T[:, 5]
        r = dict(thr=thr, hz=hz, kL=kL, kS=kS, mod=mode, k4=k4, ts=ts, tH=tH, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp98=np.percentile(dd, 98),
                 R2=np.corrcoef(np.arange(len(x30)), x30)[0, 1] ** 2, blokMed=np.median(bp), blokPoz=(bp > 0).mean() * 100, maeP95=np.percentile(T[:, 6], 95) * 100, maeMax=T[:, 6].max() * 100,
                 top3=np.sort(tp)[-3:].sum() / eq[-1] * 100, short=tp[T[:, 2] < 0].sum())
        for nm, (a_, b_) in WIN.items(): r[nm] = eq[b_] - eq[a_]
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/HOLD_1DK.csv', index=False); pd.set_option('display.width', 270)
print(R.sort_values('net', ascending=False).head(12).round(1).to_string(index=False))
G = R[(R.IS > 0) & (R.OOS > 0) & (R.zayıf > 0)]; print('\nİki yarı + zayıf pozitif, BoB benzeri (net/DDp98 × R² × blokPoz):')
G = G.assign(skor=G.net / G.DDp98 * G.R2 * G.blokPoz / 100); print(G.sort_values('skor', ascending=False).head(8).round(1).to_string(index=False))
