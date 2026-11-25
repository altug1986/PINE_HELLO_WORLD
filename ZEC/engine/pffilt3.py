src = open('/home/claude/zecwork/pffilt.py').read(); exec(src.split("rows = [ev(")[0])
dirn = TR[:, 2].astype(int); F = np.ones(n, np.bool_); kd = np.flatnonzero(dirn == -1); pu = pnl[kd]; xu = xi[kd]; eu = ei[kd]
for jj in range(len(kd)):
    prev = np.flatnonzero(xu[:jj] <= eu[jj])[-5:]
    if len(prev) == 5: F[eu[jj]] = pf(pu[prev]) >= 0.9
e, nt, pfr, inm = follow(o, c, TICK, vpos, vxp, vnew, F, 3); r = ev(e, nt, pfr, inm, v='short PF son 5 ≥ 0.9')
r['SC4'] = sc4(S(e[::10], ix[::10]), 500, True)['SC4']; print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()})
R = pd.read_pickle('/home/claude/zecwork/pffilt.pkl'); q = R[(R.filtre == 'Super-Calmar son 60 gün') & (R.eşik == -0.5) & (R.yön == 'iyiyken işlem') & (R['mod'] == 2)]
print(q[['net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD95', 'DD99', 'Ulcer', 'Calmar', 'işlem']].round(2).to_string(index=False))
print(R.iloc[[0]][['DD95']].round(1).to_string(index=False), R[(R.filtre == 'PF son 90 gün') & (R.eşik == 0.7) & (R['mod'] == 1) & (R.yön == 'iyiyken işlem')][['DD95']].round(1).to_string(index=False))
