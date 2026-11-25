# LONG TASFİYE DÖNÜŞ yan motoru — Pine'a sadık saatlik sürüm + sağlamlık: komşu plato, OI gecikmesi, maymun (rastgele FLAT anı long), yıllar
exec(open('/home/claude/zecwork/zecfikir2.py').read().split("base = ev(eq0")[0])
base = ev(eq0, K, 1.5, 1, v='V2.9')
ixh = pd.to_datetime(tms, unit='ms'); hr = pd.DataFrame({'c': c, 'oi': oi}, ixh).resample('1h').last()
def evH(zt, ot, lagOI=0):
    oih = hr.oi if lagOI == 0 else S(oi, ixh).shift(lagOI).resample('1h').last()
    r = np.log(hr.c).diff(); sd = r.rolling(720).std(ddof=0); z = r / sd; oc = oih / oih.shift(1) - 1
    hit = (z <= zt) & (oc <= ot); tend = hr.index[hit.fillna(False).values] + pd.Timedelta('1h')   # saat KAPANINCA bilinir → bir sonraki saatin ilk dakikası
    m = np.zeros(n, np.bool_); j = np.searchsorted(ixh.values, tend.values); m[j[j < n]] = True; return m
rows = []
for zt, ot, H, stp in itertools.product((-2.5, -3.0, -3.5, -4.0), (-0.01, -0.02, -0.03, -0.04), (720, 1440, 2880), (0.03, 0.05, 0.08)):
    e, ns = followE(o, h, l, c, TICK, vpos, vxp, evH(zt, ot), H, stp, 1); rows.append(ev(e, ns, 0, 0, z=zt, OI=ot, H=H, stop=stp))
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - base[k]
R['ikiYarı'] = (R.dIS > 0) & (R.dOOS > 0); R.to_pickle('/home/claude/zecwork/zecfikir3.pkl'); pd.set_option('display.width', 260)
print(f"SAATLİK (Pine'a sadık) {len(R)} varyant | artı %{(R.dnet > 0).mean() * 100:.0f} | iki yarı %{R.ikiYarı.mean() * 100:.0f} | medyan dNet {R.dnet.median():.0f} | kapı (≥{base['net'] * 1.06:.0f} & MaxDD≤391,34 & Ulcer≤26,11 & iki yarı): {((R.net >= base['net'] * 1.06) & (R.MaxDD <= 391.35) & (R.Ulcer <= 26.11) & R.ikiYarı).sum()}")
print(R.pivot_table(index=['z'], columns=['OI'], values='dnet', aggfunc='median').round(0).to_string()); print(R.pivot_table(index=['H'], columns=['stop'], values='dnet', aggfunc='median').round(0).to_string())
c_ = ['z', 'OI', 'H', 'stop', 'işlem', 'net', 'dnet', 'dIS', 'dOOS', 'yakın', 'son300', 'MaxDD', 'DD95', 'DD99', 'Ulcer', 'Calmar']
R['minY'] = np.minimum(R.dIS, R.dOOS); print('İLK YARIDA SEÇİLEN 6:'); print(R.sort_values('dIS', ascending=False)[c_].head(6).round(2).to_string(index=False))
print('İKİ YARIDA EN İYİ 6:'); print(R.sort_values('minY', ascending=False)[c_].head(6).round(2).to_string(index=False))
