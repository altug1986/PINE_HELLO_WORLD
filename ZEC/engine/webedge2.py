# Web aileleri → V2.9 üstünde doğrudan kural testi: (a/b) 12 s / 4 s net taker akışına KARŞI girişi veto, (c) emir defteri ±%1 veto, (d) funding ödeme öncesi veto, (e) baz FLAT iken akış yönünde dolgu
src = open('/home/claude/zecwork/pffilt.py').read(); exec(src.split("rows = [ev(")[0])
import glob
M = pd.concat([pd.read_parquet(f, columns=['volume', 'taker_buy_base', 'book_m1', 'book_p1', 'funding_pct']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))], ignore_index=True)
v_ = M.volume.values; tb_ = np.nan_to_num(M.taker_buy_base.values); ct_ = np.cumsum(2 * tb_ - v_); cv_ = np.cumsum(v_)
def flow(W): g = np.full(n, np.nan); g[W:] = (ct_[W:] - ct_[:-W]) / np.maximum(cv_[W:] - cv_[:-W], 1e-9); return g
FL = {240: flow(240), 720: flow(720)}; d1 = (M.book_m1.values - M.book_p1.values) / (M.book_m1.values + M.book_p1.values); fr = M.funding_pct.values
mm_ = (tms // 60000) % 60; hh_ = (tms // 3600000) % 24; dv = TR[:, 2].astype(int); isS = ei < mid; base = ev(eq0, K, 1.5, 1, v='V2.9')
def runF(lab, veto):
    Fv = np.ones(n, np.bool_); Fv[ei[veto]] = False; e, nt, pf_, im = follow(o, c, TICK, vpos, vxp, vnew, Fv, 3)
    return ev(e, nt, pf_, im, v=lab, veto=int(veto.sum()), vetoNet=pnl[veto].sum(), vetoIS=pnl[veto & isS].sum(), vetoOOS=pnl[veto & ~isS].sum())
rows = []
for W in (240, 720):
    s_ = FL[W][ei] * dv
    for q in (0.1, 0.2, 0.3): rows.append(runF(f'{W // 60} s net taker akışına karşı giriş veto (alt %{int(q * 100)})', np.nan_to_num(s_ <= np.nanquantile(s_[isS], q), nan=False)))
g_ = d1[ei] * dv; okb = np.isfinite(g_)
for q in (0.1, 0.2, 0.3): rows.append(runF(f'emir defteri ±%1 giriş yönünde kalabalık veto (üst %{int(q * 100)})', okb & (np.nan_to_num(g_) >= np.nanquantile(g_[isS & okb], 1 - q))))
mins_to = (60 - mm_[ei]) % 60 + 60 * ((8 - (hh_[ei] % 8) - 1) % 8); pay = np.r_[np.nan, fr[:-1]][ei] * dv
for th in (0.01, 0.03): rows.append(runF(f'funding ödemesine <60 dk ve giriş yönü funding öder (>{th}%) veto', (mins_to < 60) & (np.nan_to_num(pay) > th)))
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - base[k]
# (e) baz FLAT iken 12 s akış yönünde dolgu, 12 s tut
@njit(cache=False)
def followB(o, c, tick, vpos, vxp, sig, H):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; mb = 0; isB = False; eq = np.zeros(n); nb = 0
    for i in range(1, n):
        if real != 0 and not isB and real == vpos[i - 1] and vxp[i] > 0: rz += q * (vxp[i] - ep) * real - 0.0004 * q * (ep + vxp[i]); real = 0
        v = vpos[i]
        if v != 0: tg = v
        elif isB and i - mb < H: tg = real
        else:
            tg = sig[i]
            if isB and tg == real and tg != 0: mb = i
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0; isB = False
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if v == 0: isB = True; mb = i; nb += 1
        elif v != 0: isB = False
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nb
f12 = FL[720]; flat = vpos == 0; a_ = np.abs(f12[:mid][flat[:mid] & np.isfinite(f12[:mid])])
for x in (0.05, 0.10, 0.20):
    th = np.quantile(a_, 1 - x); sig = np.where(flat & (np.abs(np.nan_to_num(f12)) >= th), np.sign(np.nan_to_num(f12)), 0).astype(np.int64)
    e, nb = followB(o, c, TICK, vpos, vxp, sig, 720); r = ev(e, nb, 0, 0, v=f'baz FLAT iken 12 s akış yönünde dolgu (üst %{int(x * 100)})', veto=nb)
    for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): r['d' + k] = r[k] - base[k]
    R = pd.concat([R, pd.DataFrame([r])], ignore_index=True)
pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 70)
print(R[['v', 'veto', 'vetoNet', 'vetoIS', 'vetoOOS', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']].round(1).to_string(index=False))
R.to_csv('/home/claude/zecwork/webedge2.csv', index=False)
