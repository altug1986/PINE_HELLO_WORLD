# V3.0 motor kontrolü — Pine ifadesinin birebir karşılığı: oi = base28 'oi' (TV ZECUSDT.P_OI coin, kapanmış 5 dk), dOiV = (oi·close)/(oi[60]·close[60]) − 1
exec(open('/home/claude/zecwork/zecfikir4.py').read().split("rows = []")[0])
from sc4 import sc4
oiT = S(Z['oi'].astype(float)).ffill().values; lv = np.log(oiT * c); dOiV = np.exp(lagd(lv, 60)) - 1
m = np.nan_to_num((zr <= -3.5) & (dOiV <= -0.02), nan=False).astype(np.bool_); e1, n1 = followE(o, h, l, c, TICK, vpos, vxp, m, 1440, 0.08, 1)
r = ev(e1, n1, 0, 0); b = base
for k in ('net', 'IS', 'OOS', 'yakın', 'son300', 'MaxDD', 'DD95', 'DD99', 'Ulcer', 'Calmar'): print(f'{k}: baz {b[k]:.2f} → V3.0 {r[k]:.2f} ({r[k] - b[k]:+.2f})')
x = e1[::30]; d_ = np.maximum.accumulate(x) - x; x0 = eq0[::30]; d0 = np.maximum.accumulate(x0) - x0; print(f'DD P98: {np.percentile(d0, 98):.1f} → {np.percentile(d_, 98):.1f} | yan tetik {n1}')
rng = np.random.default_rng(3)
def mc(e, nb=3000):
    w = S(e[::30], ix[::30]).resample('W').last().diff().dropna().values; out = []
    for _ in range(nb):
        y = np.cumsum(w[rng.integers(0, len(w) - 4, 13)[:, None] + np.arange(4)].ravel()); out.append((np.maximum.accumulate(np.r_[0, y]) - np.r_[0, y]).max())
    return np.percentile(out, [90, 95, 98, 99]).round(0)
print('12 ay ileri MaxDD P90/P95/P98/P99 baz', mc(eq0), '→ V3.0', mc(e1)); print('SC4', round(sc4(S(eq0[::10], ix[::10]), 500, True)['SC4'], 4), '→', round(sc4(S(e1[::10], ix[::10]), 500, True)['SC4'], 4))
yr = S(e1 - eq0, ix).resample('YE').last(); print('yıllık ek $:', dict(zip(yr.index.year, np.diff(np.r_[0, yr.values]).round(0))))
w0 = S(eq0[::30], ix[::30]).resample('W').last().diff().dropna(); w1 = S(e1[::30], ix[::30]).resample('W').last().diff().dropna()
print(f'haftalık Sharpe {w0.mean() / w0.std() * np.sqrt(52):.2f} → {w1.mean() / w1.std() * np.sqrt(52):.2f} | Sortino {w0.mean() / w0[w0 < 0].std() * np.sqrt(52):.2f} → {w1.mean() / w1[w1 < 0].std() * np.sqrt(52):.2f} | pozitif hafta %{(w0 > 0).mean() * 100:.0f} → %{(w1 > 0).mean() * 100:.0f}')
h0 = np.diff(eq0[::60]); h1_ = np.diff(e1[::60]); print(f'1 s std $ {h0.std():.2f} → {h1_.std():.2f} | 10 dk std $ {np.diff(eq0[::10]).std():.2f} → {np.diff(e1[::10]).std():.2f}')
np.save('/home/claude/zecwork/eq_V30.npy', e1)
