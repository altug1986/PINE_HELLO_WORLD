exec(open('/home/claude/zecwork/maske2.py').read().split("rows = []")[0])
S, A = SYS['4 s SuperTrend 10/3']; e = comb(o_, h_, l_, cc, 0.01, rp, rx, rn, S, A, 5, 35.0, 0.05); r = met(e, 'V3.4')
print({k: round(v, 2) for k, v in r.items() if k != 'v'}, '| taban', {k: round(B[k], 2) for k in ('net', 'MaxDD', 'DD99', 'Ulcer', 'Calmar')})
x = pd.Series(e - e0, ixh); y = x.resample('YE').last(); print('ek katkı yıllık:', dict(zip(y.index.year, np.diff(np.r_[0, y.values]).round(0))))
act = (rp == 0) & (S != 0) & (A >= 35); print(f'ek sistem devrede olabileceği süre %{act.mean() * 100:.1f} (V3.3 FLAT + 4 s ADX≥35)')
t23 = np.searchsorted(tms, 1675382400000); print(f'TV penceresi: V3.3 {e0[-1] - e0[t23]:.1f} → V3.4 {e[-1] - e[t23]:.1f}')
for zz in (30.0, 35.0, 40.0):
    for nm in ('4 s SuperTrend 10/3', '4 s EMA10/30', '4 s EMA20/50', '4 s Donchian 20/10'):
        Sx, Ax = SYS[nm]; ee = comb(o_, h_, l_, cc, 0.01, rp, rx, rn, Sx, Ax, 5, zz, 0.05); m = met(ee, '')
        print(f'ADX≥{zz:.0f} {nm}: {m["net"] - B["net"]:+.0f} (yarılar {m["IS"] - B["IS"]:+.0f}/{m["OOS"] - B["OOS"]:+.0f}) MaxDD {m["MaxDD"]:.0f}')
