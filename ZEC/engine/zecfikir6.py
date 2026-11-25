# Son seçim: SADECE FİYAT (TV'de dış veri yok) vs coin cinsinden OI (gerçek kontrat azalması) + MC 12 ay DD + SC4 + maymun
exec(open('/home/claude/zecwork/zecfikir4.py').read().split("rows = []")[0])
from sc4 import sc4
oz = np.log(S(pd.concat([pd.read_parquet(f, columns=['oi_zec']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))], ignore_index=True).oi_zec.values).ffill().values)
rng = np.random.default_rng(11)
def mc(e, nb=2000):
    w = S(e[::30], ix[::30]).resample('W').last().diff().dropna().values; out = []
    for _ in range(nb):
        x = np.cumsum(w[rng.integers(0, len(w) - 4, 13)[:, None] + np.arange(4)].ravel()); out.append((np.maximum.accumulate(np.r_[0, x]) - np.r_[0, x]).max())
    return np.percentile(out, [90, 95, 99]).round(0)
rows = []
for zt, H, stp in itertools.product((-3.0, -3.25, -3.5, -3.75, -4.0), (720, 1440), (0.05, 0.08)):
    m = np.nan_to_num(zr <= zt, nan=False).astype(np.bool_); e, ns = followE(o, h, l, c, TICK, vpos, vxp, m, H, stp, 1); rows.append(ev(e, ns, 0, 0, tip='sadece fiyat', z=zt, OI=np.nan, H=H, stop=stp))
for zt, ot in itertools.product((-3.0, -3.5), (-0.01, -0.02, -0.03)):
    m = np.nan_to_num((zr <= zt) & (lagd(oz, 60) <= ot), nan=False).astype(np.bool_); e, ns = followE(o, h, l, c, TICK, vpos, vxp, m, 1440, 0.08, 1); rows.append(ev(e, ns, 0, 0, tip='coin OI azalması', z=zt, OI=ot, H=1440, stop=0.08))
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - base[k]
pd.set_option('display.width', 260); print(R[['tip', 'z', 'OI', 'H', 'stop', 'işlem', 'net', 'dnet', 'dIS', 'dOOS', 'yakın', 'son300', 'MaxDD', 'DD95', 'DD99', 'Ulcer', 'Calmar']].round(2).to_string(index=False))
m = np.nan_to_num(zr <= -3.5, nan=False).astype(np.bool_); e1, n1 = followE(o, h, l, c, TICK, vpos, vxp, m, 1440, 0.08, 1)
fl = np.flatnonzero((vpos == 0) & np.isfinite(SDH)); mk = []
for _ in range(1000):
    mm_ = np.zeros(n, np.bool_); mm_[rng.choice(fl, size=int(n1 * 1.6), replace=False)] = True; e, ns = followE(o, h, l, c, TICK, vpos, vxp, mm_, 1440, 0.08, 1); mk.append((e[-1] - eq0[-1]) * n1 / max(ns, 1))
mk = np.array(mk); print(f"SEÇİLEN sadece fiyat z≤−3,5 24 s stop %8: +{e1[-1] - eq0[-1]:.0f} $ | maymun ort {mk.mean():+.0f} P99 {np.percentile(mk, 99):+.0f} → p={(mk >= e1[-1] - eq0[-1]).mean():.3f}")
yr = S(e1 - eq0, ix).resample('YE').last(); print('YILLIK EK KÂR $:', dict(zip(yr.index.year, np.diff(np.r_[0, yr.values]).round(0))))
print('12 AY İLERİ MaxDD P90/P95/P99 — baz', mc(eq0), '| seçilen', mc(e1)); print('SC4 baz', round(sc4(S(eq0[::10], ix[::10]), 500, True)['SC4'], 4), '| seçilen', round(sc4(S(e1[::10], ix[::10]), 500, True)['SC4'], 4))
x = e1[::30]; d_ = np.maximum.accumulate(x) - x; print(f"seçilen DD P98 {np.percentile(d_, 98):.0f} | baz DD P98 {np.percentile(np.maximum.accumulate(eq0[::30]) - eq0[::30], 98):.0f}")
np.save('/home/claude/zecwork/eq_V30_fiyat.npy', e1); R.to_csv('/home/claude/zecwork/V30_secim.csv', index=False)
