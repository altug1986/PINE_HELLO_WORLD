exec(open('/home/claude/zecwork/adx4.py').read().split("SETS = [")[0])
M4 = np.nanmean(np.vstack([AX[k] for k in ('ZEC', 'BTC', 'XMR', 'ETH')]), 0); out = []
def ztf_adx(m):   # ZEC ADX başka dilimde (komşu dilim testi)
    D = pd.DataFrame({'high': h_, 'low': l_, 'close': cc}, ixh).resample(f'{m}min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna(); ax = dmi(D.high.values, D.low.values, D.close.values)
    ct = D.index.values + np.timedelta64(m, 'm'); j = np.searchsorted(ct, ixh.values, side='right') - 1; return np.where(j >= 0, ax[np.maximum(j, 0)], np.nan)
ZX = {5: AX['ZEC'], 3: ztf_adx(3), 10: ztf_adx(10), 15: ztf_adx(15)}
for tf, zt, m4, tm, mode in itertools.product((3, 5, 10, 15), (10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0), (99.0, 25.0, 20.0, 18.0, 15.0), (0, 1), (0, 2)):
    if tf != 5 and (m4 != 99.0 or tm == 1): continue
    z = ZX[tf] * (fH[hr] if tm else 1.0); rest = (np.nan_to_num(z, nan=99.0) < zt) & (np.nan_to_num(M4, nan=99.0) < m4)
    e, nb = ovR(o_, cc, 0.01, fp, fx, rest, mode); m = met(e, '')
    out.append((tf, zt, m4, tm, ['yeni giriş yok', '', 'geç giriş yok'][mode], m['net'] - B5['net'], m['IS'] - B5['IS'], m['OOS'] - B5['OOS'], m['MaxDD'], m['DD99'], m['Calmar'], nb, rest.mean() * 100))
d = pd.DataFrame(out, columns=['ZECdk', 'ZECadx<', '4lüADX<', 'saat', 'mod', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar', 'engel', 'dinlenme%']); d.to_csv('adx4b.csv', index=False)
pd.set_option('display.width', 250); g = (d.dIS > 0) & (d.dOOS > 0) & (d.MaxDD <= B5['MaxDD'])
print('varyant', len(d), '| iki yarı + & MaxDD ≤ baz', int(g.sum()), '| kapı', int((g & (d.dNet >= 0.06 * B5['net'])).sum()))
x = d[(d.ZECdk == 5) & (d.saat == 0) & (d['4lüADX<'] == 99.0)]; print(x.round(1).to_string(index=False))
print('--- komşu dilim (geç giriş yok)'); print(d[(d['mod'] == 'geç giriş yok') & (d['4lüADX<'] == 99.0) & (d.saat == 0)].pivot(index='ZECadx<', columns='ZECdk', values='dNet').round(0).to_string())
print('--- 4 parite ile çapraz (5 dk, geç giriş yok, saat yok)'); print(d[(d.ZECdk == 5) & (d['mod'] == 'geç giriş yok') & (d.saat == 0)].pivot(index='ZECadx<', columns='4lüADX<', values='dNet').round(0).to_string())
print(d[(d.ZECdk == 5) & (d['mod'] == 'geç giriş yok') & (d.saat == 0)].pivot(index='ZECadx<', columns='4lüADX<', values='MaxDD').round(0).to_string())
print('--- saat çarpanlı (5 dk, geç giriş yok, 4lü yok)'); print(d[(d.ZECdk == 5) & (d['mod'] == 'geç giriş yok') & (d.saat == 1) & (d['4lüADX<'] == 99.0)][['ZECadx<', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99']].round(1).to_string(index=False))
