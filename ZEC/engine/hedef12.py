# HEDEF 12.000: motor × yön katkısı, zayıf motor-yön kapatma (ablasyon), motor başına tutma süresi (TH) ızgarası — taban V3.3
exec(open('/home/claude/zecwork/kuyruk.py').read().split("rows = []")[0])
def ch(XX, THx, TSx=None):
    eqb, Tb = simY(*args, *XX, TS6 if TSx is None else TSx, THx, spk, 60, 3, byp, 0.10, 0.04, 1, False, False, False)
    P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
    for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
    vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR = np.zeros((20000, 4)); run29(args[0], args[1], args[2], cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR)
    e, _, _ = f3E(args[0], args[1], args[2], cc, 0.01, vp, vx, vn, np.ones(n, np.bool_), EVT, 1440, 0.08, np.zeros(n, np.int64), 0, 0.0); return e, Tb
e0, T0 = ch(XV, THv); B = met(e0, 'V3.3'); print(f"V3.3 {e0[-1]:.1f} | TH {THv.tolist()} TS {TS6.tolist()}", flush=True)
ad = {1: 'E0', 2: 'E1', 3: 'E2', 4: 'E3', 5: 'POMPA'}; own = T0[:, 7].astype(int); dv = T0[:, 2]; mi = T0[:, 0] < midI
K_ = pd.DataFrame([dict(motor=ad.get(k, k), yön='long' if d > 0 else 'short', adet=((own == k) & (dv == d)).sum(), net=T0[(own == k) & (dv == d), 5].sum(), ilk=T0[(own == k) & (dv == d) & mi, 5].sum(), ikinci=T0[(own == k) & (dv == d) & ~mi, 5].sum()) for k in range(1, 6) for d in (1, -1)])
print('orkestra (gölge) sahip motor × yön:'); print(K_.round(0).to_string(index=False), flush=True)
rows = [B]
for k in range(1, 5):
    for d, (ia, ie) in ((1, (0, 2)), (-1, (1, 3))):
        X2 = [x.copy() for x in XV]; X2[ia][k] = False; X2[ie][k] = False; e, _ = ch(X2, THv); rows.append(met(e, f'{ad[k]} {"long" if d > 0 else "short"} kapalı')); print(rows[-1]['v'], round(e[-1], 1), flush=True)
for k in range(1, 5):
    for mlt in (0.5, 2.0):
        T2 = THv.copy(); T2[k] = int(THv[k] * mlt); e, _ = ch(XV, T2); rows.append(met(e, f'{ad[k]} TH {THv[k]} → {T2[k]}')); print(rows[-1]['v'], round(e[-1], 1), flush=True)
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - B['net' if k == 'net' else k] if k != 'net' else R[k] - B['net']
pd.set_option('display.width', 250); print(R[['v', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']].round(1).to_string(index=False)); R.to_csv('/home/claude/zecwork/hedef12.csv', index=False)
