# Yön takibi-2: veriler "yukarı itki son dönemde sürüyor, aşağı itki her iki yarıda dönüyor" diyor → SADECE LONG yön yakalama, üst rejim şartıyla (4 s ara trend / günlük EMA20 üstü)
exec(open('/home/claude/zecwork/yontakip.py').read().split("_, _, _, _, pa60 = liq(")[0])
stt = Z['stt'].astype(float) > 0; Dd = S(c, ixh).resample('1D').last(); emd = Dd.ewm(span=20, adjust=False).mean(); up_d = (Dd > emd).astype(float)
s_ = S(up_d.values, up_d.index + pd.Timedelta('1D')); jj = np.searchsorted(s_.index.values, ixh.values, side='right') - 1; DE = np.where(jj >= 0, s_.values[np.maximum(jj, 0)], 0) > 0.5
REG = {'4 s ara trend long': stt, 'günlük kapanış > EMA20': DE, 'rejim şartı yok': np.ones(n, np.bool_)}
for N in (120, 240, 480):
    hh = S(h).rolling(N).max().shift(1).values; lo = S(l).rolling(N).min().values
    for m, rg, k, mode in itertools.product((0.03, 0.05, 0.08), REG, (6.0, 10.0, 15.0), (1, 3)):
        sL = np.nan_to_num((c > hh) & (c / lo - 1 >= m), nan=False) & REG[rg]
        e, nP, pP = ovT(o, h, l, c, TICK, rp, rx, rn, sL, Z0, A60, k, 1440, mode)
        rows.append(ev(e, nP, 0, 0, N=N, itki=m, rejim=rg, iz=k, mod='üstün' if mode == 1 else 'short veto'))
R = pd.DataFrame(rows)
for k_ in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k_] = R[k_] - B[k_]
R['ikiYarı'] = (R.dIS > 0) & (R.dOOS > 0); R.to_pickle('/home/claude/zecwork/yontakip2.pkl'); pd.set_option('display.width', 300)
print(f"{len(R)} varyant | iki yarı artı {R.ikiYarı.sum()} | kapı: {((R.net >= B['net'] * 1.06) & (R.MaxDD <= B['MaxDD'] + 0.01) & (R.Ulcer <= B['Ulcer']) & R.ikiYarı).sum()}")
print(R.groupby(['rejim', 'mod']).agg(medNet=('dnet', 'median'), maxNet=('dnet', 'max'), medIS=('dIS', 'median'), medOOS=('dOOS', 'median'), ikiYarı=('ikiYarı', 'sum')).round(0).to_string())
print(R.sort_values('dnet', ascending=False)[['N', 'itki', 'rejim', 'iz', 'mod', 'işlem', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']].head(6).round(2).to_string(index=False))
