# LONG TASFİYE DÖNÜŞ — Pine'da yapılabilir KAYAN sürüm: z = ln(close/close[60]) / (son 720 saatlik 1 s getiri std'si, tamamlanmış saatlerden), OI 60 dk değişimi 1 dk OI'den
exec(open('/home/claude/zecwork/zecfikir2.py').read().split("base = ev(eq0")[0])
base = ev(eq0, K, 1.5, 1, v='V2.9'); ixh = pd.to_datetime(tms, unit='ms')
hc = S(c, ixh).resample('1h').last(); sdh = np.log(hc).diff().rolling(720).std(ddof=0)
s_ = S(sdh.values, sdh.index + pd.Timedelta('1h')); jj = np.searchsorted(s_.index.values, ixh.values, side='right') - 1; SDH = s_.values[np.maximum(jj, 0)]; SDH[jj < 0] = np.nan
r60 = lagd(lc, 60); zr = r60 / SDH
def evR(zt, ot, lagOI=0):
    oo = loi if lagOI == 0 else np.r_[np.full(lagOI, np.nan), loi[:-lagOI]]
    return np.nan_to_num((zr <= zt) & (lagd(oo, 60) <= ot), nan=False).astype(np.bool_)
rows = []
for zt, ot, H, stp in itertools.product((-2.5, -3.0, -3.5, -4.0), (-0.01, -0.02, -0.03, -0.04), (720, 1440, 2880), (0.03, 0.05, 0.08)):
    e, ns = followE(o, h, l, c, TICK, vpos, vxp, evR(zt, ot), H, stp, 1); rows.append(ev(e, ns, 0, 0, z=zt, OI=ot, H=H, stop=stp))
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - base[k]
R['ikiYarı'] = (R.dIS > 0) & (R.dOOS > 0); R['minY'] = np.minimum(R.dIS, R.dOOS); R.to_pickle('/home/claude/zecwork/zecfikir4.pkl'); pd.set_option('display.width', 260)
gate = (R.net >= base['net'] * 1.06) & (R.MaxDD <= 391.35) & (R.Ulcer <= 26.11) & R.ikiYarı
print(f"KAYAN (Pine'da yapılabilir) {len(R)} varyant | artı %{(R.dnet > 0).mean() * 100:.0f} | iki yarı %{R.ikiYarı.mean() * 100:.0f} | medyan dNet {R.dnet.median():.0f} | kapı: {gate.sum()}")
print(R.pivot_table(index='z', columns='OI', values='dnet', aggfunc='median').round(0).to_string()); print(R.pivot_table(index='H', columns='stop', values='dnet', aggfunc='median').round(0).to_string())
c_ = ['z', 'OI', 'H', 'stop', 'işlem', 'net', 'dnet', 'dIS', 'dOOS', 'yakın', 'son300', 'MaxDD', 'DD95', 'DD99', 'Ulcer', 'Calmar']
print('İLK YARIDA SEÇİLEN 6:'); print(R.sort_values('dIS', ascending=False)[c_].head(6).round(2).to_string(index=False))
print('KAPIYI GEÇENLER:'); print(R[gate].sort_values('dnet', ascending=False)[c_].head(10).round(2).to_string(index=False))
