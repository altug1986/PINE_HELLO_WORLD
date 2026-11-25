# ETH ADX (4 s DMI) + ETH balina vekili (1 s hacim z ≥ 2.5 × mum yönü) → yeni oylar; V2.7 tabanı, IS seçim
exec(open('/home/claude/zecwork/ls1.py').read().split("out = []\ndef ad(")[0])
E = pd.read_csv('/home/claude/peer/eth/ETHUSDT_1h.csv.gz'); E.index = pd.to_datetime(E.time, unit='ms'); E = E[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].astype(float)
def map1m(s, lagH):   # s: kapanmış bar değeri, bar açılış zamanı etiketli; lagH saat sonra kullanılabilir
    s = s.copy(); s.index = s.index + pd.Timedelta(hours=lagH); j = np.searchsorted(s.index.values, idx.values, side='right') - 1; v = s.values[np.maximum(j, 0)].astype(float); v[j < 0] = np.nan; return v
def dmi(df, n_=14):
    up = df.high.diff(); dn = -df.low.diff(); pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    rma = lambda x: pd.Series(x, df.index).ewm(alpha=1 / n_, adjust=False).mean(); atr_ = rma(tr.values)
    pdi = 100 * rma(pdm) / atr_; mdi = 100 * rma(mdm) / atr_; dx = 100 * (pdi - mdi).abs() / (pdi + mdi); return rma(dx.values), pdi, mdi
E4 = E.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
ax, pdi, mdi = dmi(E4); ethAdx = pd.Series(np.where(ax >= 20, np.sign(pdi - mdi), 0.0), E4.index)
V['E1_ethAdx'] = np.nan_to_num(map1m(ethAdx, 4)).astype(np.float32)
ax1, pdi1, mdi1 = dmi(E); V['E2_ethAdx1s'] = np.nan_to_num(map1m(pd.Series(np.where(ax1 >= 25, np.sign(pdi1 - mdi1), 0.0), E.index), 1)).astype(np.float32)
qz = (E.quote_volume - E.quote_volume.rolling(720, min_periods=240).mean()) / E.quote_volume.rolling(720, min_periods=240).std()
wh = pd.Series(np.where(qz >= 2.5, np.sign(E.close - E.open), 0.0), E.index)
V['E3_ethBalina1s'] = np.nan_to_num(map1m(wh, 1)).astype(np.float32)
wh4 = wh.replace(0, np.nan).ffill(limit=3).fillna(0); V['E4_ethBalina4s'] = np.nan_to_num(map1m(wh4, 1)).astype(np.float32)
for k in ('E1_ethAdx', 'E2_ethAdx1s', 'E3_ethBalina1s', 'E4_ethBalina4s'): print(k, 'oy oranı %', round((V[k] != 0).mean() * 100, 1), '· long/short', int((V[k] > 0).sum() / 60), int((V[k] < 0).sum() / 60), flush=True)
out = []
def ad(nm, eq, T):
    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum()
    i300 = np.searchsorted(idx.values, w300); t23a = p('2023-01-01'); y = eq[t23a::30] - eq[t23a]; y3 = eq[i300::30] - eq[i300]
    yb = np.r_[0, eq[np.searchsorted(idx.values, pd.date_range('2022-01-01', '2026-01-01', freq='YS').values)], eq[-1]]
    r = dict(model=nm, n=len(T), net=eq[-1], PF=gp / gl, MaxDD=dd.max(), DDp99=np.percentile(dd, 99), IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], TVpen=eq[-1] - eq[t23],
             R2_2023p=np.corrcoef(np.arange(len(y)), y)[0, 1] ** 2, son300=y3[-1], R2_300=np.corrcoef(np.arange(len(y3)), y3)[0, 1] ** 2, yıllar=' '.join(f'{v:.0f}' for v in np.diff(yb)))
    cn, ck, kp, kr, kg = portm(eq, w300); r['çuk300$'] = cn
    out.append(r); print(pd.DataFrame([r]).round(2).to_string(index=False, header=len(out) == 1), flush=True); return eq, T
def runC(CFx):
    E_ = []; z = np.zeros(n, np.bool_)
    for od, thr, hz, kL, kS, useB, cS, w, wf, thX, cLm in CFx:
        L, S = selF(R0i, thr, hz, od); aL = L >= kL; aS = (S >= kS) & ~aL; eL, eS = gates(L, S, kL, kS, useB, cS, w, wf, 0.0, thX, cLm); E_.append((aL, aS, eL, eS))
    X = st(*([(z, z, z, z)] + E_ + [(pmp, z, pmp, z)]))
    return z03.simN8(o, h, l, c, a, *X, TS6, TH6, spk, 60, 3, byp, 0.10, False, F0, F0)
eq, T = runC(CF); ad('V2.7 baz', eq, T)
SETS = {'ETH ADX 4s': ['E1_ethAdx'], 'ETH ADX 1s': ['E2_ethAdx1s'], 'ETH balina 1s': ['E3_ethBalina1s'], 'ETH balina 4s': ['E4_ethBalina4s'],
        'ETH ADX 4s + balina 1s': ['E1_ethAdx', 'E3_ethBalina1s'], 'ETH hepsi': ['E1_ethAdx', 'E2_ethAdx1s', 'E3_ethBalina1s', 'E4_ethBalina4s']}
for nm, ex in SETS.items():
    for where in ('tüm motorlar', 'yalnız E3'):
        CFx = [((od + ex) if (where == 'tüm motorlar' or i == 3) else od, *rest) for i, (od, *rest) in enumerate(CF)]
        eq, T = runC(CFx); ad(f'+{nm} ({where})', eq, T); np.save(f"/home/claude/zecwork/eq_ETH_{nm.replace(' ', '')}_{where[:3]}.npy", eq[::30])
pd.DataFrame(out).to_csv('/home/claude/pine_hello_world/ZEC/research/ETH_ADX_BALINA_1DK.csv', index=False)
