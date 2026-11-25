exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
order = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h', 'P12_oiLiq', 'P11_oiCont']; sel.cache_clear()
L, S = sel(0, 25, 480); t23 = p('2023-02-06'); out = []
for nm, kS in (('V1.1 (short ≥2 oy)', 2), ('short ≥3 oy', 3), ('short ≥4 oy', 4), ('yalnız long', 99)):
    aL = L >= 2; aS = (S >= kS) & ~aL; eL, eS = gates(L, S, 2, kS, 1, 0, 10, 0.4, 0.0, 4.0, 2)
    eq, T = z03.sim11(o, h, l, c, a, aL, aS, eL, eS, 40.0, 1920, spk, 60); x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30
    w = eq[t23:] - eq[t23]; w30 = w[::30]; dw = np.maximum.accumulate(w30) - w30; m = T[:, 0] >= t23
    out.append(dict(sürüm=nm, net_tüm=eq[-1], MaxDD_tüm=dd.max(), IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], net_2023sonrası=w[-1], MaxDD_2023sonrası=dw.max(), işlem_2023sonrası=m.sum(), short_2023sonrası=T[m & (T[:, 2] < 0), 5].sum()))
pd.set_option('display.width', 220); print(pd.DataFrame(out).round(0).to_string(index=False))
