# SHORT'TA DAYANMA YOK — KUYRUĞU KES: zarar tavanı yöne göre ayrı (short %2,5…%7, long %10 / %5–7), tüm zincir (orkestra → uçta ters → tasfiye dönüş), taban V3.2
src = open('/home/claude/zecwork/cikis.py').read(); exec(src.split("e0, T0_ = chainX(")[0])
a0 = src.index('@njit(cache=False)\ndef simX'); b0 = src.index('    return eq, T[:nt]', a0) + len('    return eq, T[:nt]')
sy = src[a0:b0].replace('def simX(', 'def simY(').replace('byp, cap, oppN', 'byp, capL, capS, oppN').replace('cap * ep', '(capL if pos == 1 else capS) * ep'); exec(sy)
def chainY(capL, capS):
    eqb, Tb = simY(*args, *XV, TS6, THv, spk, 60, 3, byp, capL, capS, 1, False, False, False)
    P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
    for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
    vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR = np.zeros((20000, 4))
    _, K_ = run29(args[0], args[1], args[2], cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR); TR = TR[:K_]
    e, nt, _ = f3E(args[0], args[1], args[2], cc, 0.01, vp, vx, vn, np.ones(n, np.bool_), EVT, 1440, 0.08, np.zeros(n, np.int64), 0, 0.0)
    sh = Tb[:, 2] < 0; return e, Tb, sh
rows = []
for capL, capS in ((0.10, 0.10), (0.10, 0.07), (0.10, 0.06), (0.10, 0.05), (0.10, 0.04), (0.10, 0.035), (0.10, 0.03), (0.10, 0.025), (0.07, 0.07), (0.05, 0.05), (0.07, 0.05), (0.05, 0.035)):
    e, Tb, sh = chainY(capL, capS); x = e[::30]; d = np.maximum.accumulate(x) - x
    r = met(e, f'long tavan %{capL * 100:g} · short tavan %{capS * 100:g}'); r['shortEnKötü'] = Tb[sh, 5].min(); r['shortNet'] = Tb[sh, 5].sum(); r['shortAdet'] = sh.sum(); r['longEnKötü'] = Tb[~sh, 5].min(); r['DD95'] = np.percentile(d, 95); rows.append(r)
    print(r['v'], round(r['net'], 1), flush=True)
R = pd.DataFrame(rows); b = R.iloc[0]
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - b[k]
pd.set_option('display.width', 300); print(R[['v', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD95', 'DD99', 'Ulcer', 'Calmar', 'shortAdet', 'shortNet', 'shortEnKötü', 'longEnKötü']].round(1).to_string(index=False)); R.to_csv('/home/claude/zecwork/kuyruk.csv', index=False)
