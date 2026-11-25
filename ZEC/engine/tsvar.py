exec(open('/home/claude/zecwork/orchrob3.py').read().split("eq, T = stack(0)")[0])
R0 = RL[0]; LS.clear()
E = [sig('B', PR + OI + ['X17_basis'], 25, 480, 2, 2, 1, 0, 10, 0.4, 4.0, 2), sig('A', PR + OI, 10, 120, 2, 1, 0, 0, 10, 0.4, 99.0, 0),
     sig('B', PR + OI + ['X17_basis'], 25, 120, 2, 1, 0, 0, 10, 0.0, 99.0, 0), sig('G', PR + OI + EXT, 25, 240, 2, 2, 0, 0, 10, 0.4, 99.0, 0)]
AL, AS, EL, ES = st(*E); t23 = p('2023-02-06'); out = []
for ts in (28.0, 40.0, 60.0, 90.0, 1e6):
    eq, T = z03.simN3(o, h, l, c, a, AL, AS, EL, ES, np.array([ts] * 4), np.array([3840, 480, 960, 480]), spk, 60)
    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; g = T[1:, 0] - T[:-1, 1]; gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum()
    out.append(dict(stop=('yok' if ts > 1e5 else f'{ts:.0f}×ATR'), n=len(T), net=eq[-1], PF=gp / gl, MaxDD=dd.max(), DDp99=np.percentile(dd, 99), IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], TVpencere=eq[-1] - eq[t23],
                    aynıYönYenidenGiriş=int(((g <= 1) & (T[1:, 2] == T[:-1, 2])).sum()), maeMaks=T[:, 6].max() * 100, maeP95=np.percentile(T[:, 6], 95) * 100))
pd.set_option('display.width', 220); print(pd.DataFrame(out).round(1).to_string(index=False))
