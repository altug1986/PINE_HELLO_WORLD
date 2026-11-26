# KUYRUK ZARAR SERİSİ KESİCİ + SAĞLIKLI DÖNÜŞ: V3.5 işlemleri üstünde. Kes: art arda N zarar & seri zararı ≥ X $ (veya 24 s gerçekleşen ≤ −X).
# Dönüş: süre (T saat, sonra YENİ sinyal) / gölge sağlık (beklerken sistemin kâğıt işlemleri: ilk kazanç, 2 art arda kazanç, gölge net ≥ +Y). En uzun bekleme 7 gün.
exec(open('/home/claude/zecwork/bekle5.py').read().split("out = []")[0])
import itertools
e_, TT = comb3T(o_, h_, l_, cc, 0.01, rp, rx, S9, A9, 38.0, 0.05); TT = TT[np.argsort(TT[:, 0], kind='stable')]
EI, XI, DR, EP, PN = TT[:, 0].astype(np.int64), TT[:, 1].astype(np.int64), TT[:, 2], TT[:, 3], TT[:, 5]
@njit(cache=False)
def eqSel(c, EI, XI, DR, EP, PN, sel, n):
    R = np.zeros(n); M = np.zeros(n)
    for k in range(len(EI)):
        if not sel[k]: continue
        q = 500.0 / c[EI[k]]
        for t in range(EI[k], XI[k]): M[t] += q * (c[t] - EP[k]) * DR[k] - 0.0004 * q * EP[k]
        R[XI[k]] += PN[k]
    return np.cumsum(R) + M
def run(N, X, mode, P, trig='seri', W24=-1e9):
    K = len(EI); sel = np.zeros(K, np.bool_); act = True; sN = 0; sL = 0.0; pst = -1; shN = 0; shW = 0; shR = 0; shP = 0.0; nCut = 0; recent = []
    for k in range(K):
        if not act:
            if mode == 'T' and EI[k] - pst >= P: act = True
            if EI[k] - pst >= 7 * 1440: act = True
        sel[k] = act
        pn = PN[k]
        if sel[k]:
            recent.append((XI[k], pn)); recent = [r for r in recent if XI[k] - r[0] < 1440]
            if pn < 0: sN += 1; sL += pn
            else: sN = 0; sL = 0.0
            cut = (sN >= N and sL <= -X) if trig == 'seri' else (sum(r[1] for r in recent) <= W24)
            if cut: act = False; pst = XI[k]; shN = shW = shR = 0; shP = 0.0; sN = 0; sL = 0.0; nCut += 1; recent = []
        else:
            shN += 1; shP += pn; shW += pn > 0; shR = shR + 1 if pn > 0 else 0
            if (mode == 'W1' and pn > 0) or (mode == 'W2' and shR >= 2) or (mode == 'Y' and shP >= P): act = True
    return sel, nCut
def ev(sel):
    e = eqSel(cc, EI, XI, DR, EP, PN, sel, n); m = met(e, ''); p = PN[sel]; s = 0.0; w = 0.0
    for v in p: s = s + v if v < 0 else 0.0; w = min(w, s)
    return m, w, e
base = np.ones(len(EI), np.bool_); mB, wB, eB = ev(base); print(f'baz V3.5 işlem toplamı {PN.sum():.1f} | eq {eB[-1]:.1f} (comb3 {e_[-1]:.1f}) | MaxDD {mB["MaxDD"]:.1f} DD99 {mB["DD99"]:.1f} | en kötü zarar serisi {wB:.1f} | işlem {len(PN)}')
RES = [('T', 360), ('T', 720), ('T', 1440), ('W1', 0), ('W2', 0), ('Y', 25.0), ('Y', 50.0)]; out = []
for (N, X), (md, P) in itertools.product([(n_, x_) for n_ in (2, 3, 4) for x_ in (0.0, 25.0, 50.0, 75.0, 100.0)], RES):
    sel, nc = run(N, X, md, P); m, w, _ = ev(sel); out.append((f'seri {N}× & ≥{X:.0f}$', md + (str(P) if md in ('T', 'Y') else ''), m['net'] - mB['net'], m['IS'] - mB['IS'], m['OOS'] - mB['OOS'], m['MaxDD'], m['DD99'], m['Calmar'], w, nc, int((~sel).sum())))
for W24, (md, P) in itertools.product((-50.0, -75.0, -100.0), RES):
    sel, nc = run(0, 0, md, P, 'w24', W24); m, w, _ = ev(sel); out.append((f'24 s ≤ {W24:.0f}$', md + (str(P) if md in ('T', 'Y') else ''), m['net'] - mB['net'], m['IS'] - mB['IS'], m['OOS'] - mB['OOS'], m['MaxDD'], m['DD99'], m['Calmar'], w, nc, int((~sel).sum())))
d = pd.DataFrame(out, columns=['kes', 'dönüş', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar', 'enKötüSeri', 'kesme', 'atlanan']); d.to_csv('kuyruk2.csv', index=False)
g = (d.dIS > 0) & (d.dOOS > 0) & (d.MaxDD <= mB['MaxDD']); print('varyant', len(d), '| pozitif', int((d.dNet > 0).sum()), '| iki yarı + & MaxDD ≤ baz', int(g.sum()), '| kapı', int((g & (d.dNet >= 0.06 * mB['net'])).sum()))
pd.set_option('display.width', 250); print('dönüş medyan', d.groupby('dönüş').dNet.median().round(0).to_dict()); print('kes medyan', d.groupby('kes').dNet.median().round(0).to_dict())
print(d.sort_values('dNet', ascending=False).head(10).round(1).to_string(index=False)); print('--- en düşük MaxDD (iki yarı +)'); print(d[(d.dIS > 0) & (d.dOOS > 0)].sort_values('MaxDD').head(5).round(1).to_string(index=False))
