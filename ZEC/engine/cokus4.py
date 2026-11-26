# HEDEF 12.000: ikinci olay motoru — 4 SAATLİK ÇÖKÜŞ TEPKİSİ (olay çalışması: 4 s ≥%8 düşüşten sonra long iki yarıda da artı, 24 s +153/+324 bps)
# V3.3 yolu üstünde: 4 s getiri ≤ −X% (veya z ≤ −k) → V3.3 FLAT (ve seçmeli short) iken long, H bar, stop
exec(open('/home/claude/zecwork/winrate.py').read().split("dur = TT[:, 1] - TT[:, 0]")[0])
lc = np.log(cc); r240 = lc - np.r_[np.full(240, np.nan), lc[:-240]]; z4 = r240 / (SDH * 2.0)
@njit(cache=False)
def ovC(o, h, l, c, tick, rp, rx, rn, sig, H, stp, mode):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); sa = False; sb = 0; ss = 0.0; blk = 0; ns = 0; sp = 0.0
    for i in range(1, n):
        if real != 0:
            xp = -1.0
            if sa and real == 1 and l[i] <= ss: xp = min(o[i], ss) - tick; sa = False; blk = sb + H
            elif (not sa) and real == rp[i - 1] and rx[i] > 0: xp = rx[i]
            if xp > 0:
                pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn; real = 0
        if sa and i - sb >= H: sa = False
        if (not sa) and sig[i] and i >= blk and ((mode == 1 and rp[i] == 0 and real == 0) or (mode == 2 and rp[i] <= 0)): sa = True; sb = i; ns += 1
        tg = 1 if sa else rp[i]
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if sa: ss = ep * (1.0 - stp)
        elif sa and i == sb and real == 1: ss = c[i] * (1.0 - stp)
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, ns
F0 = np.zeros(n, np.bool_); e0, _ = ovC(args[0], args[1], args[2], cc, 0.01, rp, rx, rn, F0, 1440, 0.08, 1); B = met(e0, 'V3.3'); print(f"V3.3 {e0[-1]:.1f}")
SIG = {f'4 s ≤ −%{x:g}': np.nan_to_num(r240 <= -x / 100, nan=False) for x in (4, 5, 6, 8, 10)}
SIG.update({f'4 s z ≤ −{k:g}': np.nan_to_num(z4 <= -k, nan=False) for k in (2.5, 3.0, 3.5)})
rows = []
for (nm, sg), H, stp, mode in itertools.product(SIG.items(), (720, 1440, 2880), (0.05, 0.08, 0.12), (1, 2)):
    e, ns = ovC(args[0], args[1], args[2], cc, 0.01, rp, rx, rn, sg.astype(np.bool_), H, stp, mode)
    rows.append(met(e, nm, H=H, stop=stp, mod='V3.3 FLAT iken' if mode == 1 else 'FLAT veya SHORT iken (short ezilir)', tetik=ns))
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - B[k]
R['ikiYarı'] = (R.dIS > 0) & (R.dOOS > 0); R.to_pickle('/home/claude/zecwork/cokus4.pkl'); pd.set_option('display.width', 300)
print(f"{len(R)} varyant | iki yarı artı {R.ikiYarı.sum()} | kapı (≥{B['net'] * 1.06:.0f}, MaxDD≤{B['MaxDD']:.1f}, Ulcer≤{B['Ulcer']:.2f}, iki yarı): {((R.net >= B['net'] * 1.06) & (R.MaxDD <= B['MaxDD'] + 0.01) & (R.Ulcer <= B['Ulcer']) & R.ikiYarı).sum()}")
print(R.groupby(['v', 'mod']).agg(medNet=('dnet', 'median'), maxNet=('dnet', 'max'), ikiYarı=('ikiYarı', 'sum')).round(0).to_string())
print(R.sort_values('dnet', ascending=False)[['v', 'H', 'stop', 'mod', 'tetik', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']].head(10).round(2).to_string(index=False))
