exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
FULL = list(order); cf = (25, 480, 2, 2, 1, 0, 10, 0.4, 0.0, 28.0, 3840, 4.0, 2)
PR = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h']; OI = ['P12_oiLiq', 'P11_oiCont']; TT = ['ttposK4h', 'ttposK1g']
SETS = {'TAM (15)': FULL, 'TV+ : fiyat + OI + top trader (10)': PR + OI + TT, 'TV : fiyat + OI (8)': PR + OI, 'TV : fiyat + top trader (8)': PR + TT,
        'tam − balina (13)': [k for k in FULL if k not in ('whaleT', 'R18_whale')], 'tam − emir defteri (14)': [k for k in FULL if k != 'R19_book'],
        'tam − basis (14)': [k for k in FULL if k != 'X17_basis'], 'tam − global L/S (14)': [k for k in FULL if k != 'R21_glb'],
        'tam − top trader (13)': [k for k in FULL if k not in TT], 'tam − OI (13)': [k for k in FULL if k not in OI]}
out = []
for nm, od in SETS.items():
    order = od; sel.cache_clear(); eq, T = run(0, *cf); x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30
    out.append(dict(set=nm, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp99=np.percentile(dd, 99), Kasım=eq[-1] - eq[p('2026-11-01')],
                    son8ay=eq[-1] - eq[p('2026-04-01')]))
pd.set_option('display.width', 200); print(pd.DataFrame(out).round(1).to_string(index=False))
