# TV-uyumlu model: fiyat + OI + basis (TV: BINANCE:ZECUSDT spot ≈ index) — hedef: TV penceresi (2023-02-06→) ve tüm dönem
exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
t23 = p('2023-02-06')
PR = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h']; OI = ['P12_oiLiq', 'P11_oiCont']
SETS = {'fiyat+OI': PR + OI, 'fiyat+OI+basis': PR + OI + ['X17_basis'], 'tam (ref)': list(order)}
rows = []; t0 = time.time()
for sn, od in SETS.items():
    order = od; sel.cache_clear()
    grid = itertools.product((15, 25), (240, 480), (2,), (2, 3), (28.0, 40.0), (1920, 3840), (0, 2), (0, 1)) if sn != 'tam (ref)' else [(25, 480, 2, 2, 28.0, 3840, 2, 0)]
    for thr, hz, kL, kS, ts, tH, cLm, cS in grid:
        eq, T = run(0, thr, hz, kL, kS, 1, cS, 10, 0.4, 0.0, ts, tH, 4.0, cLm)
        if len(T) < 300: continue
        x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; w = eq[t23:] - eq[t23]; w30 = w[::30]; dw = np.maximum.accumulate(w30) - w30
        rows.append(dict(set=sn, thr=thr, hz=hz, kS=kS, ts=ts, tH=tH, cLm=cLm, cS=cS, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp99=np.percentile(dd, 99),
                         TVpencere=w[-1], TV_MaxDD=dw.max(), TV_işlem=(T[:, 0] >= t23).sum(), short=T[T[:, 2] < 0, 5].sum()))
    print(sn, round(time.time() - t0), flush=True)
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/TV_BASIS_1DK.csv', index=False); pd.set_option('display.width', 250)
print(R[R.set == 'tam (ref)'].round(0).to_string(index=False))
f = R[(R.set != 'tam (ref)') & (R.IS > 0) & (R.OOS > 0) & (R.MaxDD <= 450)]
for sn in ('fiyat+OI', 'fiyat+OI+basis'):
    g = f[f.set == sn]; print(f'\n{sn}: en iyi TV penceresi'); print(g.sort_values('TVpencere', ascending=False).head(5).round(0).to_string(index=False)); print(' en iyi tüm dönem'); print(g.sort_values('net', ascending=False).head(3).round(0).to_string(index=False))
