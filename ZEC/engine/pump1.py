# Pompa long olay taraması: N bar getirisi ≥ eşik & hacim oranı → long, ATR iz stop + azami süre. Sabit 500 $, %0.04 + 1 tick.
import numpy as np, pandas as pd, glob, numba, itertools
d = pd.concat([pd.read_parquet(f, columns=['time', 'open', 'high', 'low', 'close', 'volume']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))]).reset_index(drop=True)
t = pd.to_datetime(d.time, unit='ms') if d.time.dtype.kind in 'iu' else pd.to_datetime(d.time)
o, h, l, c, v = (d[k].values.astype(np.float64) for k in ('open', 'high', 'low', 'close', 'volume')); n = len(c); mid = n // 2
tr = np.maximum(h - l, np.maximum(abs(h - np.r_[c[0], c[:-1]]), abs(l - np.r_[c[0], c[:-1]]))); a = pd.Series(tr).ewm(alpha=1 / 60, adjust=False).mean().values
vm = pd.Series(v).rolling(1440).mean().shift(1).values
nov = (t >= '2026-11-01').values; r60 = c / np.r_[np.full(60, np.nan), c[:-60]] - 1; j = np.nanargmax(np.where(nov, r60, np.nan)); print('Kasım en büyük 60 dk:', t[j], round(r60[j] * 100, 1), '%', 'hacim oranı', round(v[j - 5:j + 1].mean() / vm[j], 1))
@numba.njit(cache=True)
def sim(o, h, l, c, a, ev, TS, H, cd):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; st = 0.0; hi = 0.0; ei = 0; ae = 0.0; last = -10 ** 9; out = np.zeros((n // 50 + 10, 3)); m = 0
    for i in range(1, n):
        if pos:
            if l[i] <= st:
                xp = min(o[i], st) - 0.01; pnl = q * (xp - ep) - 0.0004 * q * (ep + xp); out[m, 0] = ei; out[m, 1] = i; out[m, 2] = pnl; m += 1; pos = 0
            else:
                hi = max(hi, h[i]); st = max(st, hi - TS * ae)
                if i - ei >= H:
                    xp = c[i] - 0.01; pnl = q * (xp - ep) - 0.0004 * q * (ep + xp); out[m, 0] = ei; out[m, 1] = i; out[m, 2] = pnl; m += 1; pos = 0
        if pos == 0 and ev[i] and i - last >= cd:
            ep = c[i] + 0.01; q = 500 / ep; ae = a[i]; hi = c[i]; st = ep - TS * ae; ei = i; pos = 1; last = i
    return out[:m]
rows = []
for N, th, vr in itertools.product((10, 30, 60), (0.03, 0.05, 0.08), (0, 3, 6)):
    r = c / np.r_[np.full(N, np.nan), c[:-N]] - 1; vv = pd.Series(v).rolling(N).mean().values / vm
    ev = np.nan_to_num((r >= th) & ((vv >= vr) if vr else True)).astype(np.bool_)
    for TS, H in itertools.product((4.0, 8.0, 16.0), (240, 1440)):
        T = sim(o, h, l, c, a, ev, TS, H, 240)
        if len(T) < 10: continue
        p = T[:, 2]; e = T[:, 0].astype(int); y = t[e].dt.year.values
        rows.append(dict(N=N, th=th, vr=vr, TS=TS, H=H, n=len(T), net=p.sum(), ort=p.mean(), win=(p > 0).mean(), IS=p[e < mid].sum(), OOS=p[e >= mid].sum(),
                         yılPoz=f"{sum(p[y == yy].sum() > 0 for yy in np.unique(y))}/{len(np.unique(y))}", y26=p[y == 2026].sum()))
R = pd.DataFrame(rows); pd.set_option('display.width', 250); R.to_csv('/home/claude/zecwork/POMPA_TARAMA.csv', index=False)
print(R.sort_values('net', ascending=False).head(15).round(2).to_string(index=False))
print('\nIS>0 & OOS>0:', ((R.IS > 0) & (R.OOS > 0)).sum(), '/', len(R), '· pozitif net:', (R.net > 0).sum())
