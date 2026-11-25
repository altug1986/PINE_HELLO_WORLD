# ARŞİV FİKİR 2: gölge baskı testi (x×ATR, N bar) → ulaştı: gir (mode1) / girme (mode2); ulaşamadı: TERS gir
exec(open('/home/claude/zecwork/vbar1.py').read().split("eq, T = sim(X0); ad('V2.7 baz', eq, T)")[0])
def sim10(X, th0, fN, fM, mode): return z03.simN10(o, h, l, c, a, *X, TS6, np.r_[th0, TH6[1:]], spk, 60, 3, byp, 0.10, False, F0, F0, fN, fM, mode)
eq, T = sim10(X0, 480, 0, 0.0, 1); ad('V2.7 baz', eq, T)
for mode in (1, 2):
    for fN in (3, 6, 10):
        for fM in (0.5, 1.0, 2.0):
            for th0 in (240, 960):
                eq, T = sim10(X0, th0, fN, fM, mode); r = out[-1] if False else None
                ad(f'gölge {"ulaştı→gir" if mode == 1 else "ulaştı→girme"} N{fN} x{fM} ters tut{th0}', eq, T)
                out[-1]['tersN'] = int((T[:, 7] == 0).sum()); out[-1]['ters$'] = T[T[:, 7] == 0, 5].sum()
pd.DataFrame(out).to_csv('/home/claude/pine_hello_world/ZEC/research/GOLGE_BASKI_1DK.csv', index=False)
R_ = pd.DataFrame(out); print(R_[['model', 'tersN', 'ters$']].to_string(index=False))
