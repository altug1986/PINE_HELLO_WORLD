# TV'de hesaplanabilen balina vekili: büyük hacimli 1 dk mumların yönlü akışı (TV: volume × close), gerçek balina baskısının yerine
exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
sgn = lambda x, b=0.0: np.where(x > b, 1, np.where(x < -b, -1, 0)).astype(np.float32)
qv = B['volume'].values.astype(float) * c; med = pd.Series(qv).rolling(1440, min_periods=200).median().values
PR = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h']; OI = ['P12_oiLiq', 'P11_oiCont']; TT = ['ttposK4h', 'ttposK1g']
res = []
for km in (2.0, 3.0, 5.0):
    big = (qv > km * med).astype(float); sv = np.sign(c - o) * qv * big
    V['WPX10'] = sgn(np.nan_to_num(pd.Series(sv).rolling(20).sum().values / pd.Series(qv).rolling(20).sum().values), 0.1)
    V['WPX30'] = sgn(np.nan_to_num(pd.Series(sv).rolling(60).sum().values / pd.Series(qv).rolling(60).sum().values), 0.05)
    rw = np.asarray(V['R18_whale']); wp = V['WPX10']; ag = ((rw != 0) & (wp != 0)); print(f'km={km}: gerçek balina oyu ile vekil aynı yönde: %{(rw[ag] == wp[ag]).mean() * 100:.0f} · vekil aktif %{(wp != 0).mean() * 100:.1f} / gerçek %{(rw != 0).mean() * 100:.1f}', flush=True)
    for nm, od in (('fiyat+OI+vekil', PR + OI + ['WPX10', 'WPX30']), ('fiyat+OI+vekil+toptrader', PR + OI + TT + ['WPX10', 'WPX30'])):
        order = od
        for thr, hz, kL, kS, ts, tH in itertools.product((15, 25), (240, 480), (1, 2), (1, 2), (28.0, 40.0), (1920, 3840)):
            sel.cache_clear() if False else None
            L, S = select(RL[0], thr, hz); aL = L >= kL; aS = (S >= kS) & ~aL; eL, eS = gates(L, S, kL, kS, 1, 0, 10, 0.4, 0.0, 4.0, 2)
            eq, T = z03.sim11(o, h, l, c, a, aL, aS, eL, eS, ts, tH, spk, 60)
            if len(T) < 400: continue
            x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30
            res.append(dict(km=km, set=nm, thr=thr, hz=hz, kL=kL, kS=kS, ts=ts, tH=tH, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp99=np.percentile(dd, 99), Kasım=eq[-1] - eq[p('2026-11-01')]))
R = pd.DataFrame(res); R.to_csv('/home/claude/pine_hello_world/ZEC/research/TV_BALINA_VEKIL_1DK.csv', index=False); pd.set_option('display.width', 220)
f = R[(R.IS > 0) & (R.OOS > 0)]; print(f.groupby(['set', 'km']).net.max().round(0).to_string())
print(f.sort_values('net', ascending=False).head(12).round(1).to_string(index=False))
