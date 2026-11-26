# TV'yi buradan çalıştıramıyorum → TV V3.2 export'unu motorla işlem işlem eşleştir, V3.3 için TV beklentisini kalibre et
exec(open('/home/claude/zecwork/kuyruk.py').read().split("rows = []")[0])
def real(capL, capS):
    eqb, Tb = simY(*args, *XV, TS6, THv, spk, 60, 3, byp, capL, capS, 1, False, False, False)
    P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
    for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
    vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR = np.zeros((20000, 4))
    run29(args[0], args[1], args[2], cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR)
    e, _, _ = f3E(args[0], args[1], args[2], cc, 0.01, vp, vx, vn, np.ones(n, np.bool_), EVT, 1440, 0.08, np.zeros(n, np.int64), 0, 0.0)
    rp, rx, rn = rec30(args[0], args[1], args[2], cc, 0.01, vp, vx, vn, EVT, 1440, 0.08); return e, rp, rn
T = pd.read_csv('/home/claude/tvx/v32.csv'); g = T[T.Tip.str.contains('giriş')].copy(); g['t'] = pd.to_datetime(g['Tarih ve saat']) - pd.Timedelta('3h'); g['d'] = np.where(g.Tip.str.contains('Uzun'), 1, -1)
t0 = g.t.min(); i0 = np.searchsorted(ixh.values, t0.to_datetime64()); tE = pd.Timestamp('2026-11-23 23:59')
for lab, cL, cS in (('V3.2', 0.10, 0.10), ('V3.3', 0.10, 0.04)):
    e, rp, rn = real(cL, cS); ent = np.flatnonzero(rn & (np.arange(n) >= i0)); et = ixh[ent]; ed = rp[ent]
    gw = g[g.t <= tE]; ek = pd.Series(ed, et)
    m = sum(1 for t, d in zip(gw.t, gw.d) if ((ek.index >= t - pd.Timedelta('2min')) & (ek.index <= t + pd.Timedelta('2min')) & (ek.values == d)).any())
    print(f"{lab} motor: TV penceresi ({t0:%Y-%m-%d} → 2026-11-23) net {e[np.searchsorted(ixh.values, tE.to_datetime64())] - e[i0]:.1f} $ · giriş {len(ent)} | TV giriş {len(gw)} · eşleşen (±2 dk, aynı yön) {m} = %{m / len(gw) * 100:.1f}")
pn = g[g.t <= tE]['Net K/Z USDT'].astype(float); print(f"TV V3.2 aynı pencere net {pn.sum():.1f} $")
