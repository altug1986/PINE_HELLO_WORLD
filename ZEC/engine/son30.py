exec(open('/home/claude/zecwork/pompa8.py').read().split("out = []; t0 = time.time()")[0])
ok1 = np.ones(n, np.bool_); z0 = np.zeros(n, np.bool_); zf = np.zeros(n)
e35 = comb3(o_, h_, l_, cc, 0.01, rp, rx, S9, A9, 38.0, 0.05)[0]
rpW, rxW, _ = chainW(20.0, 0.03); e36 = combR(o_, h_, l_, cc, 0.01, rpW, rxW, S9, A9, 38.0, 0.05, z0, zf, zf, 720, 240, 0.0, 0, 0.05, ok1)[0]
E = pd.DataFrame({'V3.3': e0, 'V3.5': e35, 'V3.5+ADX çıkış': e36}, index=ixh); t1 = E.index[-1]; t0 = t1 - pd.Timedelta('30D')
X = E.loc[t0:] - E.loc[t0].values; print('dönem', t0, '→', t1)
x30 = X.iloc[::30]; dd = (x30.cummax() - x30).max(); print('net', X.iloc[-1].round(1).to_dict(), '| MaxDD', dd.round(1).to_dict())
dly = X.resample("1D").last(); print("GÜN", [str(i.date())[5:] for i in dly.index]); print("V33", dly["V3.3"].round(0).astype(int).tolist()); print("V35", dly["V3.5"].round(0).astype(int).tolist())
