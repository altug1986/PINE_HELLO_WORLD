# 5 GÜNLÜK KÂR KONTROLÜ: her P günde bir son P günün KAPANMIŞ işlem neti < eşik ise 2–3 saat bekle (V3.5 üstüne)
exec(open('/home/claude/zecwork/sync26.py').read().split("e, TT = comb3T(")[0])
@njit(cache=False)
def comb3P(o, h, l, c, tick, rp, rx, S, A, athr, capP):
    n = len(c); real = 0; fill = False; cap = 0.0; fp = np.zeros(n, np.int8); fx = np.zeros(n)
    for i in range(1, n):
        if real != 0:
            xp = -1.0
            if fill and ((real == 1 and l[i] <= cap) or (real == -1 and h[i] >= cap)): xp = (min(o[i], cap) - tick) if real == 1 else (max(o[i], cap) + tick)
            elif (not fill) and real == rp[i - 1] and rx[i] > 0: xp = rx[i]
            if xp > 0: fx[i] = xp; real = 0; fill = False
        v = rp[i]; s = int(S[i]); tg = v
        if tg == 0 and s != 0 and A[i] >= athr: tg = s
        nf = tg != 0 and v == 0
        if tg != real:
            real = tg
            if tg != 0 and nf: cap = (c[i] + tick * tg) * (1.0 - capP * tg)
        elif nf and not fill: cap = c[i] * (1.0 - capP * tg)
        fill = nf; fp[i] = real
    return fp, fx
@njit(cache=False)
def ovP(o, c, tick, fp, fx, P, thr, pause, mode):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); pend = False; pEnd = -1; nP = 0; nC = 0; rzH = np.zeros(n)
    for i in range(1, n):
        if real != 0 and real == fp[i - 1] and fx[i] > 0:
            rz += q * (fx[i] - ep) * real - 0.0004 * q * (ep + fx[i]); real = 0
        rzH[i] = rz; tg = int(fp[i])
        if i % P == 0 and i >= P:
            nC += 1
            if rz - rzH[i - P] < thr:
                nP += 1
                if mode == 1: pend = True
                else: pEnd = i + pause
        if mode == 1 and pend and (real == 0 or tg != real): pEnd = i + pause; pend = False   # mevcut işlem kapanınca bekleme başlar
        if pEnd >= i:
            if mode == 2 and real != 0 and tg == real: pass                                   # açık işlem sürer, yeni giriş yok
            else: tg = 0
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
            if tg != 0: real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nP, nC
fp, fx = comb3P(o_, h_, l_, cc, 0.01, rp, rx, S9, A9, 38.0, 0.05)
e0b, _, _ = ovP(o_, cc, 0.01, fp, fx, 7200, -1e9, 0, 0); eR = comb3(o_, h_, l_, cc, 0.01, rp, rx, S9, A9, 38.0, 0.05)[0]
B5 = met(e0b, ''); print(f'baz V3.5 (uzatılmış veri): {B5["net"]:.1f} (comb3 {eR[-1]:.1f}) MaxDD {B5["MaxDD"]:.1f} DD99 {B5["DD99"]:.1f} Calmar {B5["Calmar"]:.2f}')
out = []
for Pd in (4, 5, 6):
    for thr in (0.0, -10.0, -25.0, -50.0):
        for pause in (120, 180):
            for mode in (0, 1, 2):
                e, nP, nC = ovP(o_, cc, 0.01, fp, fx, Pd * 1440, thr, pause, mode); m = met(e, '')
                out.append((Pd, thr, pause, ['hemen kapat', 'pozisyon kapanınca', 'sadece yeni giriş yok'][mode], m['net'] - B5['net'], m['IS'] - B5['IS'], m['OOS'] - B5['OOS'], m['MaxDD'], m['DD99'], m['Calmar'], nP, nC))
d = pd.DataFrame(out, columns=['gün', 'eşik', 'bekle', 'mod', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar', 'bekleme', 'kontrol']); d.to_csv('bekle5.csv', index=False)
pd.set_option('display.width', 220); print(d[d['gün'] == 5].round(1).to_string(index=False)); print(d.groupby('mod').dNet.median().round(0).to_dict())
print('gün bazında medyan Δnet', d.groupby('gün').dNet.median().round(0).to_dict(), '| pozitif varyant', int((d.dNet > 0).sum()), '/', len(d), '| iki yarı + & MaxDD ≤ baz', int(((d.dIS > 0) & (d.dOOS > 0) & (d.MaxDD <= B5['MaxDD'])).sum()))
print(d.sort_values('dNet', ascending=False).head(5).round(1).to_string(index=False))
