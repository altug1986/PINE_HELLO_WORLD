# Ara trend motoru ile orkestra arasında birim paylaşımı: (a) trend en düşük öncelik (b) trend durumu yalnız short vetosu
exec(open('/home/claude/zecwork/trend2.py').read().split("eq, T = sim(X0); ad('V2.7 baz', eq, T); out[-1]")[0])
def simLow(stt):
    z = np.zeros(n, np.bool_); tl = stt > 0
    rows = [(z, z, z, z)] + [tuple(X0[q][r] for q in range(4)) for r in range(1, 5)] + [(tl, z, tl, z), tuple(X0[q][5] for q in range(4))]
    X = [np.ascontiguousarray(np.vstack([r_[q] for r_ in rows])) for q in range(4)]
    return z03.simN11(o, h, l, c, a, *X, np.r_[1e6, TS6[1:5], 1e6, 8.0], np.r_[480, TH6[1:5], 1, 480], spk, 60, 3, np.array([0, 0, 0, 0, 0, 1, 1], np.bool_), 0.10, False, F0, F0, np.r_[0.0, [0.10] * 4, 0.0, 0.10])
eq, T = sim(X0); ad('V2.7 baz', eq, T); out[-1].update(ralliHariçNet=eq[-1] - (eq[r1] - eq[r0]))
for tp in (0.15, 0.25, 0.40):
    for k in (4, 6):
        stt = state2(200, 55, 20, tp, k); nm = f'EMA200 kırılım55/20 kâr%{int(tp * 100)} iz{k}'
        eq, T = simLow(stt); ad('V2.7 + trend EN ALT öncelik: ' + nm, eq, T); out[-1].update(extra(eq, T, 5))
        X = [x.copy() for x in X0]; X[1][1:5] &= ~(stt > 0)[None, :]; X[3][1:5] &= ~(stt > 0)[None, :]
        eq, T = sim(X); ad('V2.7 + trend yalnız short VETO: ' + nm, eq, T); out[-1].update(ralliHariçNet=eq[-1] - (eq[r1] - eq[r0]))
R = pd.DataFrame(out); R.to_csv('/home/claude/pine_hello_world/ZEC/research/ARA_TREND_BIRIM_1DK.csv', index=False)
cols = ['model', 'n', 'net', 'PF', 'MaxDD', 'DDp99', 'IS', 'OOS', 'son300', 'R2_2023p', 'trendN', 'trend', 'ralliHariçNet', 'yıllar']
pd.set_option('display.width', 320); print(R[[c_ for c_ in cols if c_ in R]].round(1).to_string(index=False))
