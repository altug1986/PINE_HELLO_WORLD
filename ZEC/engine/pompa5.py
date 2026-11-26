# POMPA long stop tabanı: stop mesafesi = min(max(8×ATR, taban × giriş), tavan × giriş) — sadece POMPA (motor 5)
exec(open('/home/claude/zecwork/maske10.py').read().split("bn = B['net']; out = []")[0])
src2 = open('/home/claude/zecwork/cikis.py').read(); a0 = src2.index('@njit(cache=False)\ndef simX'); b0 = src2.index('    return eq, T[:nt]', a0) + len('    return eq, T[:nt]')
sz = src2[a0:b0].replace('def simX(', 'def simZ(').replace('byp, cap, oppN', 'byp, capL, capS, flP, oppN').replace('cap * ep', '(capL if pos == 1 else capS) * ep')
for v in ('j', 'kk', 'k'): sz = sz.replace(f'TS[{v}] * ae', f'max(TS[{v}] * ae, (flP if {v} == 5 else 0.0) * ep)')
exec(sz); S9, A9 = mkm(190)
def chainZ(flP):
    eqb, Tb = simZ(*args, *XV, TS6, THv, spk, 60, 3, byp, 0.10, 0.04, flP, 1, False, False, False)
    P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
    for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
    vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR = np.zeros((20000, 4)); run29(args[0], args[1], args[2], cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR)
    rp_, rx_, rn_ = rec30(args[0], args[1], args[2], cc, 0.01, vp, vx, vn, EVT, 1440, 0.08)
    e = comb3(o_, h_, l_, cc, 0.01, rp_, rx_, S9, A9, 38.0, 0.05)[0]; pm = Tb[Tb[:, 7] == 5]; return e, pm
R = {}
for fl in (0.0, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08):
    e, pm = chainZ(fl); m = met(e, ''); R[fl] = m
    print(f'taban %{fl * 100:4.1f}: net {m["net"]:8.1f} yarılar {m["IS"]:7.1f}/{m["OOS"]:7.1f} MaxDD {m["MaxDD"]:5.1f} DD99 {m["DD99"]:5.1f} Ulcer {m["Ulcer"]:4.1f} Calmar {m["Calmar"]:.2f} | POMPA {len(pm)} işlem net {pm[:, 5].sum():+6.1f} kazanma %{(pm[:, 5] > 0).mean() * 100:.0f} en kötü {pm[:, 5].min():.1f}', flush=True)
