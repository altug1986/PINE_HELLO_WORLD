# ARA TREND: rejim (4s EMA üstü + eğim) içinde Donchian kırılımla gir, kâr al (%) / kısa kanal / ATR izle çık, yeni kırılımda tekrar gir (her giriş 500 $)
exec(open('/home/claude/zecwork/trend1.py').read().split("eq, T = sim(X0); ad('V2.7 baz', eq, T)\nfor N, M")[0])
r0, r1 = p('2025-09-01'), p('2026-01-01')
def extra(eq, T, own):
    m = T[:, 7] == own; x = T[m, 5]; tot = x.sum()
    return dict(trendN=int(m.sum()), trend=tot, enBüyükPay=(x.max() / tot * 100) if tot > 0 else np.nan, ralliHariçNet=eq[-1] - (eq[r1] - eq[r0]))
def state2(E, Ne, Nx, tp, k):
    C = Z4.c.values; em = Z4.c.ewm(span=E, adjust=False).mean().values; reg = (C > em) & (em > np.r_[np.full(6, np.nan), em[:-6]])
    hi = Z4.h.rolling(Ne).max().shift(1).values; lo = Z4.l.rolling(Nx).min().shift(1).values; st_ = np.zeros(len(C)); s = 0; ep = ext = 0.0
    for t in range(len(C)):
        if s == 1:
            ext = max(ext, C[t])
            if (not reg[t]) or C[t] < lo[t] or (k > 0 and C[t] < ext - k * A4[t]) or (tp > 0 and C[t] >= ep * (1 + tp)): s = 0
        elif reg[t] and C[t] > hi[t]: s = 1; ep = ext = C[t]
        st_[t] = s
    return np.nan_to_num(map1m(pd.Series(st_, Z4.index), 4))
eq, T = sim(X0); ad('V2.7 baz', eq, T); out[-1].update(ralliHariçNet=eq[-1] - (eq[r1] - eq[r0]))
for E in (100, 200):
    for Ne, Nx in ((20, 10), (55, 20)):
        for tp in (0.15, 0.25, 0.40, 0.0):
            for k in (4, 6):
                nm = f'EMA{E} kırılım{Ne}/{Nx} kâr%{int(tp * 100)} iz{k}'; stt = state2(E, Ne, Nx, tp, k)
                eq, T = simT(stt, False); ad('TEK: ' + nm, eq, T); out[-1].update(extra(eq, T, 1))
                eq, T = simT(stt, True); ad('V2.7+' + nm, eq, T); out[-1].update(extra(eq, T, 1))
R = pd.DataFrame(out); R.to_csv('/home/claude/pine_hello_world/ZEC/research/ARA_TREND_1DK.csv', index=False); pd.set_option('display.width', 300)
cols = ['model', 'n', 'net', 'PF', 'MaxDD', 'DDp99', 'IS', 'OOS', 'son300', 'R2_2023p', 'trendN', 'trend', 'enBüyükPay', 'ralliHariçNet']
print('\n=== BİRLEŞİK, MaxDD ≤ 500 ==='); print(R[R.model.str.startswith('V2.7') & (R.MaxDD <= 500)].sort_values('net', ascending=False)[cols].head(12).round(1).to_string(index=False))
print('\n=== BİRLEŞİK, net sırası ==='); print(R[R.model.str.startswith('V2.7')].sort_values('net', ascending=False)[cols].head(8).round(1).to_string(index=False))
