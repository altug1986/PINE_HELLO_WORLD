exec(open('/home/claude/zecwork/maske2.py').read().split("rows = []")[0])
bn = B['net']; out = []
S1, _ = SYS['1 s EMA10/30']
for nm in ('4 s SuperTrend 10/3', '4 s EMA10/30', '4 s EMA20/50'):
    Sx, Ax = SYS[nm]
    for uy in (0, 1):
        Su = np.where((S1 == Sx) | (uy == 0), Sx, 0)
        for zz in (35.0, 40.0, 45.0):
            for st in (0.03, 0.04, 0.05):
                m = met(comb(o_, h_, l_, cc, 0.01, rp, rx, rn, Su, Ax, 5, zz, st), '')
                out.append((nm + (' +1s uyum' if uy else ''), zz, st, m['net'] - bn, m['IS'] - B['IS'], m['OOS'] - B['OOS'], m['MaxDD'], m['DD99'], m['Calmar']))
d = pd.DataFrame(out, columns=['sis', 'adx', 'stop', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar']); d.to_csv('maske5.csv', index=False)
g = d[(d.dNet >= 0.06 * bn) & (d.MaxDD <= B['MaxDD']) & (d.dIS > 0) & (d.dOOS > 0)]
print('kapı geçen:', len(g), '/', len(d)); print(d.sort_values('Calmar', ascending=False).head(12).round(1).to_string(index=False))
