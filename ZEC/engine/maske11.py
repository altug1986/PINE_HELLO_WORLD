exec(open('/home/claude/zecwork/maske10.py').read().split("bn = B['net']; out = []")[0])
bn = B['net']; out = []
for m in range(150, 290, 10):
    S, A = mkm(m)
    for z in range(34, 45):
        e, nE, _ = comb3(o_, h_, l_, cc, 0.01, rp, rx, S, A, float(z), 0.05); r = met(e, '')
        out.append((m, z, r['net'] - bn, r['IS'] - B['IS'], r['OOS'] - B['OOS'], r['MaxDD'], r['DD99'], r['Ulcer'], r['Calmar'], nE))
d = pd.DataFrame(out, columns=['dk', 'adx', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar', 'giris']); d.to_csv('maske11.csv', index=False)
d['kapi'] = (d.dNet >= 0.06 * bn) & (d.MaxDD <= B['MaxDD']) & (d.dIS > 0) & (d.dOOS > 0)
P = d.pivot(index='dk', columns='adx', values='dNet'); K = d.pivot(index='dk', columns='adx', values='kapi').astype(float)
nb = P.rolling(3, center=True, min_periods=2).median().T.rolling(3, center=True, min_periods=2).median().T   # ±10 dk × ±1 ADX komşu medyanı
kb = K.rolling(3, center=True, min_periods=2).mean().T.rolling(3, center=True, min_periods=2).mean().T
d['komsuNet'] = [nb.loc[a, b] for a, b in zip(d.dk, d.adx)]; d['komsuKapi'] = [kb.loc[a, b] for a, b in zip(d.dk, d.adx)]
print('kapı geçen', int(d.kapi.sum()), '/', len(d)); print(d[d.kapi].sort_values('komsuNet', ascending=False).head(10).round(2).to_string(index=False))
print(K.astype(int).replace({1: '#', 0: '.'}).to_string())
