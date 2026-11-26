exec(open('/home/claude/zecwork/maske2.py').read().split("rows = []")[0])
for rule, lab in (('2h', '2 s'), ('3h', '3 s'), ('6h', '6 s'), ('8h', '8 s'), ('12h', '12 s')):
    td = pd.Timedelta(rule); D = bars(rule); C = D.c; ax = m1(adx(D), D, td)
    for f, s in ((10, 30), (20, 50)):
        SYS[f'{lab} EMA{f}/{s}'] = (m1(np.sign(C.ewm(span=f, adjust=False).mean() - C.ewm(span=s, adjust=False).mean()).values, D, td), ax)
    H, L = D.h.values, D.l.values; Cv = C.values; pc = np.r_[Cv[0], Cv[:-1]]; at = rma(np.maximum(H - L, np.maximum(abs(H - pc), abs(L - pc))), 10); hl = (H + L) / 2; up = hl - 3 * at; dn = hl + 3 * at
    st = np.zeros(len(D)); fu, fd, d_ = up[0], dn[0], 1
    for t in range(1, len(D)):
        fu = max(up[t], fu) if Cv[t - 1] > fu else up[t]; fd = min(dn[t], fd) if Cv[t - 1] < fd else dn[t]
        d_ = 1 if Cv[t] > fd else (-1 if Cv[t] < fu else d_); st[t] = d_
    SYS[f'{lab} SuperTrend 10/3'] = (m1(st, D, td), ax)
bn = B['net']; out = []
for lab in ('1 s', '2 s', '3 s', '4 s', '6 s', '8 s', '12 s'):
    for sy in ('EMA10/30', 'EMA20/50', 'SuperTrend 10/3'):
        S, A = SYS[f'{lab} {sy}']
        for zz in (30.0, 35.0, 40.0):
            m = met(comb(o_, h_, l_, cc, 0.01, rp, rx, rn, S, A, 5, zz, 0.05), '')
            out.append((lab, sy, zz, m['net'] - bn, m['IS'] - B['IS'], m['OOS'] - B['OOS'], m['MaxDD'], m['DD99'], m['Calmar'], m['Ulcer']))
d = pd.DataFrame(out, columns=['tf', 'sis', 'adx', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar', 'Ulcer']); d.to_csv('maske6.csv', index=False)
g = d[(d.dNet >= 0.06 * bn) & (d.MaxDD <= B['MaxDD']) & (d.dIS > 0) & (d.dOOS > 0)]
print('baz', round(bn, 1), round(B['MaxDD'], 1), round(B['DD99'], 1), round(B['Calmar'], 2), '| kapı geçen:', len(g), '/', len(d))
print(d.groupby('tf', sort=False).agg(medNet=('dNet', 'median'), maxNet=('dNet', 'max'), minMaxDD=('MaxDD', 'min'), bestCalmar=('Calmar', 'max')).round(1).to_string())
print(d.sort_values('Calmar', ascending=False).head(10).round(1).to_string(index=False))
print(g.round(1).to_string(index=False))
