# KAZANMA ORANI > %55: V3.3 işlemlerinin kayıp yapısı + kâr kilidi (MFE ≥ a% olunca stop giriş+b%'ye) / sabit kâr al seçenekleri
exec(open('/home/claude/zecwork/tvrapor.py').read().split("T = pd.read_csv('/home/claude/tvx/v32.csv'); g = T[T.Tip.str.contains('giriş')]; t0")[0])
eqb, Tb = simY(*args, *XV, TS6, THv, spk, 60, 3, byp, 0.10, 0.04, 1, False, False, False)
P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR0 = np.zeros((20000, 4)); run29(args[0], args[1], args[2], cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR0)
rp, rx, rn = rec30(args[0], args[1], args[2], cc, 0.01, vp, vx, vn, EVT, 1440, 0.08); TT = trades(args[0], cc, rp, rx, rn)
dur = TT[:, 1] - TT[:, 0]; pn = TT[:, 5]; tip = np.where(TT[:, 2] != P_[TT[:, 0].astype(int)], 'katman (ters/yan)', 'orkestra')
D = pd.DataFrame({'pnl': pn, 'süre': pd.cut(dur, [0, 5, 30, 120, 480, 1e9], labels=['≤5 dk', '5-30 dk', '30 dk-2 s', '2-8 s', '>8 s']), 'tip': tip, 'yön': np.where(TT[:, 2] > 0, 'long', 'short')})
print(f"V3.3 tüm geçmiş: {len(pn)} işlem, kazanma %{(pn > 0).mean() * 100:.1f}, net {pn.sum():.0f}")
print(D.groupby(['süre'], observed=True).agg(adet=('pnl', 'size'), kazanma=('pnl', lambda s: (s > 0).mean() * 100), net=('pnl', 'sum'), ortKazanç=('pnl', lambda s: s[s > 0].mean()), ortKayıp=('pnl', lambda s: s[s <= 0].mean())).round(1).to_string())
print(D.groupby(['tip', 'yön']).agg(adet=('pnl', 'size'), kazanma=('pnl', lambda s: (s > 0).mean() * 100), net=('pnl', 'sum')).round(1).to_string())
@njit(cache=False)
def ovL(o, h, l, c, tick, rp, rx, rn, a, b, tp):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); mfe = 0.0; lk = False; ls = 0.0; skip = False; nw = 0; nl = 0; gp = 0.0; gl = 0.0
    for i in range(1, n):
        if real != 0:
            xp = -1.0
            if tp > 0 and ((real == 1 and h[i] >= ep * (1 + tp)) or (real == -1 and l[i] <= ep * (1 - tp))): xp = (max(o[i], ep * (1 + tp))) if real == 1 else (min(o[i], ep * (1 - tp))); skip = True
            elif lk and ((real == 1 and l[i] <= ls) or (real == -1 and h[i] >= ls)): xp = (min(o[i], ls) - tick) if real == 1 else (max(o[i], ls) + tick); skip = True
            elif real == rp[i - 1] and rx[i] > 0: xp = rx[i]
            if xp > 0:
                pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn; real = 0
                if pn > 0: nw += 1; gp += pn
                else: nl += 1; gl -= pn
        if real != 0:
            mfe = max(mfe, (h[i] / ep - 1) if real == 1 else (1 - l[i] / ep))
            if a > 0 and not lk and mfe >= a: lk = True; ls = ep * (1 + b * real)
        v = rp[i]
        if rn[i]: skip = False
        tg = 0 if skip else v
        if tg != real:
            if real != 0:
                xp = c[i] - tick * real; pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn; real = 0
                if pn > 0: nw += 1; gp += pn
                else: nl += 1; gl -= pn
            if tg != 0: real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]; mfe = 0.0; lk = False
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nw, nl, gp / max(gl, 1e-9)
rows = []
for a_, b_, tp in [(0, 0, 0)] + [(a_ / 100, b_ / 100, 0) for a_ in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0) for b_ in (0.1, 0.2, 0.3, 0.5) if b_ < a_] + [(0, 0, t / 100) for t in (1, 2, 3, 5, 8)]:
    e, nw, nl, pf = ovL(args[0], args[1], args[2], cc, 0.01, rp, rx, rn, a_, b_, tp)
    rows.append(met(e, 'V3.3' if a_ == 0 and tp == 0 else (f'kâr kilidi: MFE ≥%{a_ * 100:g} → stop giriş+%{b_ * 100:g}' if a_ else f'sabit kâr al %{tp * 100:g}'), kazanma=nw / (nw + nl) * 100, işlem=nw + nl, PF=pf))
R = pd.DataFrame(rows); b = R.iloc[0]
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - b[k]
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 50)
print(R[['v', 'kazanma', 'işlem', 'PF', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']].round(2).to_string(index=False)); R.to_csv('/home/claude/zecwork/winrate.csv', index=False)
