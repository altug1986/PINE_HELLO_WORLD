# Aday: KAYAN long tasfiye dönüşü z ≤ −3,5 & OI 60 dk ≤ −%2 → (baz FLAT ise) 24 s long, stop %8. Sağlamlık: OI gecikmesi, sadece fiyat, maymun, yıllar, yan işlem dökümü, MC 12 ay DD
exec(open('/home/claude/zecwork/zecfikir4.py').read().split("rows = []")[0])
ZT, OT, H, ST = -3.5, -0.02, 1440, 0.08
def run(m): e, ns = followE(o, h, l, c, TICK, vpos, vxp, m, H, ST, 1); return e, ns
e1, n1 = run(evR(ZT, OT)); r1 = ev(e1, n1, 0, 0); print(f"ADAY net {r1['net']:.1f} (+{r1['net'] - base['net']:.1f}) IS {r1['IS'] - base['IS']:+.1f} OOS {r1['OOS'] - base['OOS']:+.1f} | yan tetik {n1}")
print('OI GECİKMESİ (dk):', {L: round(run(evR(ZT, OT, L))[0][-1] - eq0[-1], 1) for L in (0, 5, 10, 15, 30)})
pz = {zt: round(run(np.nan_to_num(zr <= zt, nan=False).astype(np.bool_))[0][-1] - eq0[-1], 1) for zt in (-3.0, -3.5, -4.0)}; print('SADECE FİYAT (OI şartı yok) z:', pz)
fl = np.flatnonzero((vpos == 0) & np.isfinite(SDH)); rng = np.random.default_rng(7); mk = []
for _ in range(1000):
    m = np.zeros(n, np.bool_); m[rng.choice(fl, size=int(n1 * 1.6), replace=False)] = True; e, ns = run(m); mk.append((e[-1] - eq0[-1]) * n1 / max(ns, 1))
mk = np.array(mk); print(f"MAYMUN (rastgele FLAT anında {n1} kez 24 s long, aynı stop): ort {mk.mean():+.0f} $, P95 {np.percentile(mk, 95):+.0f} $, P99 {np.percentile(mk, 99):+.0f} $ | aday {e1[-1] - eq0[-1]:+.0f} $ → p = {(mk >= e1[-1] - eq0[-1]).mean():.3f}")
yr = S(e1 - eq0, ix).resample('YE').last().diff(); yr.iloc[0] = (e1 - eq0)[np.searchsorted(ix.values, yr.index[0].to_datetime64(), side='right') - 1]
print('YILLIK EK KÂR $:', {d.year: round(v, 0) for d, v in yr.items()})
m = evR(ZT, OT); st = np.flatnonzero(m); sa_end = -1; tr_ = []
for s0 in st:
    if s0 <= sa_end or vpos[s0] != 0: continue
    ep = c[s0] + TICK; stp = ep * (1 - ST); xi_ = min(s0 + H, n - 1); hit = np.flatnonzero(l[s0 + 1:xi_ + 1] <= stp)
    if len(hit): xx = s0 + 1 + hit[0]; xp = min(o[xx], stp) - TICK
    else: xx = xi_; xp = c[xx] - TICK
    q = 500 / c[s0]; tr_.append((ix[s0], q * (xp - ep) - 0.0004 * q * (ep + xp), (l[s0 + 1:xx + 1].min() / ep - 1) * 100)); sa_end = s0 + H
T_ = pd.DataFrame(tr_, columns=['zaman', 'pnl', 'MAE%']); print(f"YAN İŞLEM (yaklaşık, tek başına): {len(T_)} | kazanma %{(T_.pnl > 0).mean() * 100:.0f} | ort {T_.pnl.mean():+.1f} $ | PF {T_.pnl[T_.pnl > 0].sum() / -T_.pnl[T_.pnl < 0].sum():.2f} | en kötü {T_.pnl.min():.1f} $ | MAE medyan {T_['MAE%'].median():.1f}% P95 {T_['MAE%'].quantile(0.05):.1f}% | yıl başına adet {T_.groupby(T_.zaman.dt.year).size().to_dict()}")
def mc(e, nb=2000):
    w = S(e[::30], ix[::30]).resample('W').last().diff().dropna().values; out = []
    for _ in range(nb):
        x = np.cumsum(w[rng.integers(0, len(w) - 4, 13)[:, None] + np.arange(4)].ravel()); out.append((np.maximum.accumulate(np.r_[0, x]) - np.r_[0, x]).max())
    return np.percentile(out, [90, 95, 99])
print('12 AY İLERİ MaxDD (haftalık blok MC, P90/P95/P99) baz:', mc(eq0).round(0), 'aday:', mc(e1).round(0))
from sc4 import sc4
print('SC4 baz', round(sc4(S(eq0[::10], ix[::10]), 500, True)['SC4'], 4), 'aday', round(sc4(S(e1[::10], ix[::10]), 500, True)['SC4'], 4)); np.save('/home/claude/zecwork/eq_V30_aday.npy', e1); T_.to_csv('/home/claude/zecwork/V30_yan_islem.csv', index=False)
