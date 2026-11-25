exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
PR = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h']; OI = ['P12_oiLiq', 'P11_oiCont']; FULL = list(order)
C = {'V1.1 (TV: fiyat+OI)': (PR + OI, (25, 480, 2, 2, 1, 0, 10, 0.4, 0.0, 40.0, 1920, 4.0, 2)), 'B1 (TV: +basis)': (PR + OI + ['X17_basis'], (25, 480, 2, 2, 1, 0, 10, 0.4, 0.0, 28.0, 3840, 4.0, 2)),
     'B2 (TV: +basis)': (PR + OI + ['X17_basis'], (15, 480, 2, 2, 1, 0, 10, 0.4, 0.0, 28.0, 1920, 4.0, 2)), 'TAM 9.128 (API)': (FULL, (25, 480, 2, 2, 1, 0, 10, 0.4, 0.0, 28.0, 3840, 4.0, 2))}
Pp = pd.read_pickle('/home/claude/port/portfoy_canli_10dk_utc.pkl'); Pm = Pp.resample('MS').last().diff()['2023-06-01':]
wk = Pm <= Pm.quantile(0.30); out = []; MZ = {}
for nm, (od, cf) in C.items():
    order = od; sel.cache_clear(); eq, T = run(0, *cf); z = pd.Series(eq, index=idx).resample('MS').last().diff().reindex(Pm.index).fillna(0); MZ[nm] = z
    for nom in (200,):
        cm = Pm + z * nom / 500
        out.append(dict(aday=nm, net=eq[-1], zayıfAy_ZEC=z[wk].sum() * nom / 500, zayıfAyPoz=(z[wk] > 0).mean() * 100, güçlüAy_ZEC=z[~wk].sum() * nom / 500, aylıkKor=z.corr(Pm),
                        portEnKötüAy=Pm.min(), birlikteEnKötüAy=cm.min(), port_alt5=Pm.nsmallest(5).mean(), birlikte_alt5=cm.nsmallest(5).mean(), zecNegAy=(z < 0).mean() * 100))
pd.set_option('display.width', 250); print('zayıf aylar:', [f'{d:%Y-%m} {v:.0f}' for d, v in Pm[wk].items()])
print(pd.DataFrame(out).round(1).to_string(index=False))
pd.DataFrame(MZ).assign(portföy=Pm).to_csv('/home/claude/zecwork/aylik_zec_port.csv')
