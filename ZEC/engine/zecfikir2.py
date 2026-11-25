# ZEC fikir bloğu → V2.9 üstünde: (1) LONG TASFİYE DÖNÜŞ yan motoru (1 s fiyat z ≤ eşik ve OI 1 s ≤ eşik → long, H bar, stop), (2) hacim/OI rejimiyle yön vetosu
src = open('/home/claude/zecwork/pffilt.py').read(); exec(src.split("rows = [ev(")[0])
import glob
M = pd.concat([pd.read_parquet(f, columns=['quote_volume', 'oi_usdt', 'btc_close']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))], ignore_index=True)
def ff(x): return S(x).ffill().values
def lagd(x, k): return x - np.r_[np.full(k, np.nan), x[:-k]]
def zs(x, W): s = S(x); return ((s - s.rolling(W, min_periods=W // 4).mean()) / s.rolling(W, min_periods=W // 4).std()).values
lc = np.log(c); oi = ff(M.oi_usdt.values); loi = np.log(np.where(oi > 0, oi, np.nan)); oi1 = lagd(loi, 60); zp1 = zs(lagd(lc, 60), 43200); b1 = lagd(np.log(ff(M.btc_close.values)), 60)
@njit(cache=False)
def followE(o, h, l, c, tick, vpos, vxp, evt, H, stp, mode):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; sa = False; sb = 0; ss = 0.0; blk = 0; eq = np.zeros(n); ns = 0; sp = 0.0; spn = 0.0
    for i in range(1, n):
        if real != 0:
            xp = -1.0
            if sa and real == 1 and l[i] <= ss: xp = min(o[i], ss) - tick; sa = False; blk = sb + H
            elif (not sa) and real == vpos[i - 1] and vxp[i] > 0: xp = vxp[i]
            if xp > 0:
                pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn
                if xp < ep * (1 - stp * 0.5) and real == 1 and not sa: spn += 0.0
                real = 0
        if sa and i - sb >= H: sa = False
        if (not sa) and evt[i] and i >= blk and (mode != 1 or vpos[i] == 0): sa = True; sb = i; ns += 1
        tg = 1 if sa else vpos[i]
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if sa: ss = ep * (1.0 - stp)
        elif sa and i == sb: ss = ep * (1.0 - stp) if real == 1 else ss
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, ns
base = ev(eq0, K, 1.5, 1, v='V2.9'); rows = []
for zt, ot, H, stp, md in itertools.product((-2.5, -3.0, -3.5), (-0.02, -0.03, -0.04), (240, 720, 1440), (0.03, 0.05), (0, 1, 2)):
    m = np.nan_to_num((zp1 <= zt) & (oi1 <= ot), nan=False)
    if md == 2: m = m & np.nan_to_num(b1 > -0.01, nan=False)
    e, ns = followE(o, h, l, c, TICK, vpos, vxp, m.astype(np.bool_), H, stp, 1 if md == 1 else 0)
    rows.append(ev(e, ns, 0, 0, v='long tasfiye dönüş', z=zt, OI=ot, H=H, stop=stp, mod=['baz ne olursa', 'sadece baz FLAT', 'BTC 1 s > −%1 teyidi'][md]))
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - base[k]
R['ikiYarı'] = (R.dIS > 0) & (R.dOOS > 0); R['minY'] = np.minimum(R.dIS, R.dOOS); R.to_pickle('/home/claude/zecwork/zecfikir2.pkl')
pd.set_option('display.width', 260)
print(f"{len(R)} varyant | iki yarı artı %{R.ikiYarı.mean() * 100:.0f} | medyan dNet {R.dnet.median():.0f} | kapı (≥{base['net'] * 1.06:.0f}, MaxDD≤391,3, Ulcer≤26,11): {((R.net >= base['net'] * 1.06) & (R.MaxDD <= 391.34) & (R.Ulcer <= 26.11)).sum()}")
c_ = ['z', 'OI', 'H', 'stop', 'mod', 'işlem', 'net', 'dnet', 'dIS', 'dOOS', 'yakın', 'son300', 'MaxDD', 'DD95', 'DD99', 'Ulcer', 'Calmar']
print('İLK YARIDA SEÇİLEN 8:'); print(R.sort_values('dIS', ascending=False)[c_].head(8).round(2).to_string(index=False))
print('İKİ YARIDA BİRDEN EN İYİ 8:'); print(R.sort_values('minY', ascending=False)[c_].head(8).round(2).to_string(index=False))
print(R.groupby('mod').agg(medNet=('dnet', 'median'), ikiYarı=('ikiYarı', 'mean'), medDD=('dMaxDD', 'median')).round(2).to_string())
# (2) hacim/OI rejimi (D1) ile V2.9 giriş vetosu
qv = S(M.quote_volume.values).rolling(1440, min_periods=720).sum().values; D1 = zs(np.log(np.maximum(qv / np.maximum(oi, 1), 1e-9)), 43200)
dv = TR[:, 2].astype(int); isS = ei < mid; g = D1[ei]; out = []
for side, q in itertools.product(('short', 'long'), (0.1, 0.2, 0.3)):
    if side == 'short': ok = dv < 0; thr = np.nanquantile(g[isS & ok], 1 - q); veto = ok & np.nan_to_num(g >= thr, nan=False)
    else: ok = dv > 0; thr = np.nanquantile(g[isS & ok], q); veto = ok & np.nan_to_num(g <= thr, nan=False)
    Fv = np.ones(n, np.bool_); Fv[ei[veto]] = False; e, nt, pf_, im = follow(o, c, TICK, vpos, vxp, vnew, Fv, 3)
    r = ev(e, nt, pf_, im, v=f'hacim/OI {"yüksekken short" if side == "short" else "düşükken long"} veto (%{int(q * 100)})'); r['vetoNet'] = pnl[veto].sum(); r['vetoIS'] = pnl[veto & isS].sum(); r['vetoOOS'] = pnl[veto & ~isS].sum(); out.append(r)
V = pd.DataFrame(out)
for k in ('net', 'IS', 'OOS', 'MaxDD'): V['d' + k] = V[k] - base[k]
print(); print(V[['v', 'vetoNet', 'vetoIS', 'vetoOOS', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD']].round(1).to_string(index=False)); V.to_csv('/home/claude/zecwork/zecfikir2_veto.csv', index=False)
