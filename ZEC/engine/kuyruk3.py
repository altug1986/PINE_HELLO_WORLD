# SAVUNMA MODU: art arda N zarar (& seri ≥ X $) → sonraki işlemlere sıkı bıçak stop %s (giriş fiyatından); ilk kazançla (veya 2 kazanç) normal oyuna dönüş. İşlem kaçırılmaz.
exec(open('/home/claude/zecwork/kuyruk2.py').read().split("base = np.ones(")[0])
@njit(cache=False)
def defend(o, h, l, c, EI, XI, DR, EP, PN, N, X, s, back):
    K = len(EI); XI2 = XI.copy(); PN2 = PN.copy(); sN = 0; sL = 0.0; dfn = False; wins = 0; nd = 0; ns = 0
    for k in range(K):
        if dfn:
            nd += 1; d = DR[k]; st = EP[k] * (1.0 - s * d); q = 500.0 / c[EI[k]]
            for t in range(EI[k] + 1, XI[k] + 1):
                if (d == 1 and l[t] <= st) or (d == -1 and h[t] >= st):
                    xp = (min(o[t], st) - 0.01) if d == 1 else (max(o[t], st) + 0.01); p2 = q * (xp - EP[k]) * d - 0.0004 * q * (EP[k] + xp)
                    if t < XI[k] or p2 > PN[k]: XI2[k] = t; PN2[k] = p2; ns += 1
                    break
        p = PN2[k]
        if p < 0: sN += 1; sL += p; wins = 0
        else: sN = 0; sL = 0.0; wins += 1
        if dfn and wins >= back: dfn = False
        if (not dfn) and sN >= N and sL <= -X: dfn = True; wins = 0
    return XI2, PN2, nd, ns
out = []; mB, wB, eB = ev(np.ones(len(EI), np.bool_))
for N, X, s, back in itertools.product((1, 2, 3), (0.0, 25.0, 50.0), (0.01, 0.015, 0.02, 0.025, 0.03), (1, 2)):
    XI2, PN2, nd, ns = defend(o_, h_, l_, cc, EI, XI, DR, EP, PN, N, X, s, back)
    e = eqSel(cc, EI, XI2, DR, EP, PN2, np.ones(len(EI), np.bool_), n); m = met(e, ''); sr = 0.0; w = 0.0
    for v in PN2: sr = sr + v if v < 0 else 0.0; w = min(w, sr)
    out.append((N, X, s * 100, back, m['net'] - mB['net'], m['IS'] - mB['IS'], m['OOS'] - mB['OOS'], m['MaxDD'], m['DD99'], m['Calmar'], w, nd, ns))
d = pd.DataFrame(out, columns=['N', 'X', 'stop%', 'dönüşKazanç', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar', 'enKötüSeri', 'savunmaİşlem', 'kesilen']); d.to_csv('kuyruk3.csv', index=False)
g = (d.dIS > 0) & (d.dOOS > 0) & (d.MaxDD <= mB['MaxDD']); print(f'baz MaxDD {mB["MaxDD"]:.1f} DD99 {mB["DD99"]:.1f} en kötü seri {wB:.1f} | varyant', len(d), '| pozitif', int((d.dNet > 0).sum()), '| iki yarı + & MaxDD ≤ baz', int(g.sum()), '| kapı', int((g & (d.dNet >= 0.06 * mB['net'])).sum()))
pd.set_option('display.width', 250)
for k in ('N', 'X', 'stop%', 'dönüşKazanç'): print(k, d.groupby(k).dNet.median().round(0).to_dict())
print(d.sort_values('dNet', ascending=False).head(10).round(1).to_string(index=False)); print('--- iki yarı + , en düşük DD99'); print(d[(d.dIS > 0) & (d.dOOS > 0)].sort_values('DD99').head(6).round(1).to_string(index=False))
