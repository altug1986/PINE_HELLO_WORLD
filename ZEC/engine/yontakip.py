# YÖN TAKİBİ: genel eğilim (orkestra) yerine fiyat hareketi yönünü yakala ve sür. Kırılım (N bar tepesi/dibi) + itki (N bar dibinden ≥ m%) [+ salınım yapısı] → o yöne gir,
# zirveden k×ATR geri gelene kadar SÜR (kapanışa göre), %10 tavan stop, en çok 24 s. Mod: üstün (V3.0'ı ezer) / sadece ters girişi engelle.
exec(open('/home/claude/zecwork/kanama.py').read().split("F0 = np.zeros(n, np.bool_); e30, _")[0])
pc_ = np.r_[c[0], c[:-1]]; A60 = rma(np.maximum(h - l, np.maximum(np.abs(h - pc_), np.abs(l - pc_))), 60)
@njit(cache=False)
def ovT(o, h, l, c, tick, rp, rx, rn, sL, sS, A, k, H, mode):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); pd_ = 0; pb = 0; pk = 0.0; cap = 0.0; nP = 0; pP = 0.0; tr = 0; blkd = 0
    for i in range(1, n):
        if real != 0:
            xp = -1.0
            if pd_ != 0 and real == pd_ and ((real == 1 and l[i] <= cap) or (real == -1 and h[i] >= cap)): xp = (min(o[i], cap) - tick) if real == 1 else (max(o[i], cap) + tick); pd_ = 0
            elif pd_ == 0 and real == rp[i - 1] and rx[i] > 0: xp = rx[i]
            if xp > 0:
                pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn
                if tr: pP += pn
                real = 0; tr = 0
        if pd_ != 0:
            pk = max(pk, h[i]) if pd_ == 1 else min(pk, l[i])
            if (pd_ == 1 and c[i] < pk - k * A[i]) or (pd_ == -1 and c[i] > pk + k * A[i]) or i - pb >= H: pd_ = 0
        if pd_ == 0 and (sL[i] or sS[i]): pd_ = 1 if sL[i] else -1; pb = i; pk = h[i] if pd_ == 1 else l[i]
        v = rp[i]
        if mode == 1: tg = pd_ if pd_ != 0 else (v if (real == v or rn[i] or real != 0 or True) else 0)
        else:
            tg = v
            if pd_ != 0 and v == -pd_: tg = 0
        if mode == 1 and pd_ == 0 and tr == 1: tg = v
        if tg != real:
            if real != 0:
                xp = c[i] - tick * real; pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn
                if tr: pP += pn
                real = 0; tr = 0
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if mode == 1 and pd_ != 0 and tg == pd_: tr = 1; nP += 1; cap = ep * (1.0 - 0.10 * tg)
        elif mode == 1 and pd_ != 0 and real == pd_ and tr == 0: tr = 1; nP += 1; cap = ep * (1.0 - 0.10 * real)
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nP, pP
Z0 = np.zeros(n, np.bool_); e0, _, _ = ovT(o, h, l, c, TICK, rp, rx, rn, Z0, Z0, A60, 10.0, 1440, 1); print(f'V3.0 yol kontrol {e0[-1]:.1f}')
B = ev(e0, 0, 0, 0); rows = []
_, _, _, _, pa60 = liq(h, l, c, 60, 60, 0.01)
for N in (120, 240, 480):
    hh = S(h).rolling(N).max().shift(1).values; ll = S(l).rolling(N).min().shift(1).values; lo = S(l).rolling(N).min().values; hi = S(h).rolling(N).max().values
    for m, st, k, mode, side in itertools.product((0.03, 0.05, 0.08), (False, True), (6.0, 10.0, 15.0), (1, 3), ('ikisi', 'sadece long')):
        sL = np.nan_to_num((c > hh) & (c / lo - 1 >= m), nan=False) & ((pa60 > 0) if st else True); sS = np.nan_to_num((c < ll) & (1 - c / hi >= m), nan=False) & ((pa60 < 0) if st else True)
        if side == 'sadece long': sS = Z0
        e, nP, pP = ovT(o, h, l, c, TICK, rp, rx, rn, sL, sS, A60, k, 1440, mode)
        rows.append(ev(e, nP, 0, 0, N=N, itki=m, yapı=st, iz=k, mod='üstün (V3.0 ezer)' if mode == 1 else 'sadece ters girişi engelle', yön=side, yönİşlem=nP, yönKâr=pP))
R = pd.DataFrame(rows)
for k_ in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k_] = R[k_] - B[k_]
R['ikiYarı'] = (R.dIS > 0) & (R.dOOS > 0); R.to_pickle('/home/claude/zecwork/yontakip.pkl'); pd.set_option('display.width', 300)
print(f"{len(R)} varyant | iki yarı artı {R.ikiYarı.sum()} | kapı (≥{B['net'] * 1.06:.0f} & MaxDD≤{B['MaxDD']:.1f} & Ulcer≤{B['Ulcer']:.2f} & iki yarı): {((R.net >= B['net'] * 1.06) & (R.MaxDD <= B['MaxDD'] + 0.01) & (R.Ulcer <= B['Ulcer']) & R.ikiYarı).sum()}")
print(R.groupby(['mod', 'yön']).agg(medNet=('dnet', 'median'), maxNet=('dnet', 'max'), ikiYarı=('ikiYarı', 'sum'), medYönKâr=('yönKâr', 'median')).round(0).to_string())
c_ = ['N', 'itki', 'yapı', 'iz', 'mod', 'yön', 'yönİşlem', 'yönKâr', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']
print(R.sort_values('dnet', ascending=False)[c_].head(10).round(1).to_string(index=False))
