# 26 Kas senkron: uzatılmış veride V3.5 (V3.3 zinciri + 190 dk / ADX≥38 ek sistem) → işlem listesi, TV export (9 → 26 Kas) ile eşleştirme
exec(open('/home/claude/zecwork/pompa5.py').read().split("R = {}")[0])
@njit(cache=False)
def put(T, k, a, b, c_, d, e_, f, g):
    T[k, 0] = a; T[k, 1] = b; T[k, 2] = c_; T[k, 3] = d; T[k, 4] = e_; T[k, 5] = f; T[k, 6] = g
@njit(cache=False)
def comb3T(o, h, l, c, tick, rp, rx, S, A, athr, capP):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); fill = False; cap = 0.0; T = np.zeros((20000, 7)); k = 0; ei = 0; kind = 0
    for i in range(1, n):
        if real != 0:
            xp = -1.0
            if fill and ((real == 1 and l[i] <= cap) or (real == -1 and h[i] >= cap)): xp = (min(o[i], cap) - tick) if real == 1 else (max(o[i], cap) + tick)
            elif (not fill) and real == rp[i - 1] and rx[i] > 0: xp = rx[i]
            if xp > 0:
                pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn; put(T, k, ei, i, real, ep, xp, pn, kind); k += 1; real = 0; fill = False
        v = rp[i]; s = int(S[i]); tg = v
        if tg == 0 and s != 0 and A[i] >= athr: tg = s
        nf = tg != 0 and v == 0
        if tg != real:
            if real != 0: xp = c[i] - tick * real; pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn; put(T, k, ei, i, real, ep, xp, pn, kind); k += 1; real = 0
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]; ei = i; kind = 1 if nf else 0
                if nf: cap = ep * (1.0 - capP * tg)
        elif nf and not fill: cap = c[i] * (1.0 - capP * tg)
        fill = nf
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    if real != 0: put(T, k, ei, n - 1, real, ep, c[n - 1], q * (c[n - 1] - ep) * real - 0.0004 * q * (ep + c[n - 1]), kind + 10); k += 1
    return eq, T[:k]
e, TT = comb3T(o_, h_, l_, cc, 0.01, rp, rx, S9, A9, 38.0, 0.05)
i23 = np.searchsorted(tms, 1795478340000); print(f'regresyon: 23 Kas 23:59 V3.5 = {e[i23]:.1f} (eski 11.023,4) | uzatılmış son {e[-1]:.1f} @ {ixh[-1]}')
D = pd.DataFrame(TT, columns=['ei', 'xi', 'yon', 'ep', 'xp', 'pnl', 'tur']); D['giris'] = ixh[D.ei.astype(int)] + pd.Timedelta('3h'); D['cikis'] = ixh[D.xi.astype(int)] + pd.Timedelta('3h')
D = D[D.cikis >= '2026-11-09 03:00']; D.to_csv('/home/claude/zecwork/sync26_motor.csv', index=False)
t = pd.read_excel('/home/claude/tvx/v35.xlsx', sheet_name='İşlemler'); g = t[t.Tip.str.contains('giriş')].copy(); x = t[t.Tip.str.contains('çıkış')].copy()
g['t'] = pd.to_datetime(g['Tarih ve saat']); g['yon'] = np.where(g.Tip.str.contains('Uzun'), 1, -1); x = x.set_index('İşlem numarası')
V = g.set_index('İşlem numarası')[['t', 'yon', 'Fiyat USDT']].join(x[['Tarih ve saat', 'Fiyat USDT', 'Net K/Z USDT']], rsuffix='_x')
print(f"TV: {len(V)} işlem net {V['Net K/Z USDT'].sum():.1f} | motor aynı aralık (çıkış ≥ 9 Kas 03:00): {len(D)} işlem net {D.pnl.sum():.1f}")
rows = []
for k, r in V.iterrows():
    m = D[(np.abs((D.giris - r.t) / pd.Timedelta('1min')) <= 2) & (D.yon == r.yon)]
    rows.append((k, r.t.strftime('%d %H:%M'), 'L' if r.yon > 0 else 'S', r['Fiyat USDT'], str(r['Tarih ve saat'])[5:16], round(r['Net K/Z USDT'], 2),
                 (m.giris.iloc[0].strftime('%d %H:%M') + ' ' + m.cikis.iloc[0].strftime('%d %H:%M') + f' {m.pnl.iloc[0]:+.2f}' + (' ek' if m.tur.iloc[0] % 10 == 1 else '')) if len(m) else '—'))
print(pd.DataFrame(rows, columns=['#', 'TV giriş', 'y', 'fiyat', 'TV çıkış', 'TV K/Z', 'motor giriş çıkış K/Z']).to_string(index=False))
onlyM = D[[not any((np.abs((r.giris - V.t) / pd.Timedelta('1min')) <= 2) & (V.yon == r.yon)) for _, r in D.iterrows()]]
print('sadece motor:', [(r.giris.strftime('%d %H:%M'), 'L' if r.yon > 0 else 'S', round(r.pnl, 2), 'ek' if r.tur % 10 == 1 else '') for _, r in onlyM.iterrows()])
