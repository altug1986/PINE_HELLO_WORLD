# Ek: yön bazlı PF kapısı (long/short ayrı, sadece short) + seçilenlerin SUPER-CALMAR V4 (SC4) karnesi
src = open('/home/claude/zecwork/pffilt.py').read(); exec(src.split("rows = [ev(")[0])
dirn = TR[:, 2].astype(int); rows = []
for N, th, only_s in itertools.product((5, 10, 20, 30, 50), (0.7, 0.9, 1.0, 1.2), (False, True)):
    F = np.ones(n, np.bool_)
    for d_ in ((-1,) if only_s else (1, -1)):
        kd = np.flatnonzero(dirn == d_); pu = pnl[kd]; xu = xi[kd]; eu = ei[kd]
        for jj in range(len(kd)):
            prev = np.flatnonzero(xu[:jj] <= eu[jj])[-N:]
            if len(prev) == N: F[eu[jj]] = pf(pu[prev]) >= th
    rows.append(ev(*follow(o, c, TICK, vpos, vxp, vnew, F, 3), filtre=('short PF' if only_s else 'yön PF') + f' son {N} işlem', eşik=th, yön='iyiyken işlem', mod=3))
D2 = pd.DataFrame(rows); B0 = ev(*follow(o, c, TICK, vpos, vxp, vnew, np.ones(n, np.bool_), 1), filtre='YOK', eşik=0, yön='', mod=0)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): D2['d' + k] = D2[k] - B0[k]
pd.set_option('display.width', 300)
print(f"YÖN / SHORT PF KAPISI {len(D2)} varyant | iki yarı artı {((D2.dIS > 0) & (D2.dOOS > 0)).sum()} | medyan dNet {D2.dnet.median():.0f} | max dNet {D2.dnet.max():.0f} | Calmar↑ {(D2.dCalmar > 0).sum()}")
print(D2.sort_values('dnet', ascending=False)[['filtre', 'eşik', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar', 'işlem']].head(5).round(2).to_string(index=False))
# SC4: baz + ana tablodaki en iyiler (iki yarı / risk düşüren)
R = pd.read_pickle('/home/claude/zecwork/pffilt.pkl'); v = R.iloc[1:].copy(); v['minY'] = np.minimum(v.dIS, v.dOOS)
pick = pd.concat([v.sort_values('minY', ascending=False).head(2), v[(v.dDD99 < 0) & (v.dMaxDD <= 0)].sort_values('dnet', ascending=False).head(2), v[v.filtre.str.startswith('Super')].sort_values('dnet', ascending=False).head(1)])
def Fof(r):
    if r.filtre.startswith('motor'): return None
    val = MET[r.filtre]; return np.where(np.isnan(val), True, val >= r.eşik) if r.yön == 'iyiyken işlem' else np.where(np.isnan(val), True, val < r.eşik)
def S4(e):
    s = sc4(S(e[::10], ix[::10]), 500, True); return s if not isinstance(s, dict) else s.get('SC4', s)
print('\nSUPER-CALMAR V4 (ZEC tek başına, 500 $):'); out = [dict(v='YOK (V2.9)', net=B0['net'], SC4=S4(follow(o, c, TICK, vpos, vxp, vnew, np.ones(n, np.bool_), 1)[0]))]
for _, r in pick.iterrows():
    F = Fof(r)
    if F is None: continue
    e = follow(o, c, TICK, vpos, vxp, vnew, F, int(r['mod']))[0]; out.append(dict(v=f"{r.filtre} {r.yön} eşik {r.eşik:g} mod {int(r['mod'])}", net=e[-1], SC4=S4(e)))
print(pd.DataFrame(out).to_string(index=False))
