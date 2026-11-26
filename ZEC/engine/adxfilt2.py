# Ek: (a) trende TERS açılan işlemlerin geçmiş karnesi, (b) F5: trende ters girişe daha sıkı zarar tavanı (veto yerine), (c) TV V3.0'da yan motor izleri
exec(open('/home/claude/zecwork/adxfilt.py').read().split("dv = TR[:, 2].astype(int); rows = []")[0])
dv = TR[:, 2].astype(int); isS = ei < mid; ep_ = c[ei]; out = []
for tf, thr in itertools.product(('1s', '4s'), (20, 25)):
    a, p_, m_ = (x[ei] for x in PX[tf][:3]); op = np.nan_to_num((a >= thr) & (((dv < 0) & (p_ > m_)) | ((dv > 0) & (m_ > p_))), nan=False)
    for side, sm in (('short (yükseliş trendinde)', dv < 0), ('long (düşüş trendinde)', dv > 0)):
        k = op & sm; out.append(dict(trend=f'{tf} ADX≥{thr}', yön=side, adet=k.sum(), net=pnl[k].sum(), net1=pnl[k & isS].sum(), net2=pnl[k & ~isS].sum(), kazanma=(pnl[k] > 0).mean() * 100, ort=pnl[k].mean(), enKötü=pnl[k].min(), tavan10=(pnl[k] <= -49).sum()))
print(pd.DataFrame(out).round(1).to_string(index=False))
@njit(cache=False)
def f5(o, h, l, c, tick, vpos, vxp, vnew, OPP, capO, evt, H, stp):
    n = len(c); gb = 0; real = 0; ep = 0.0; q = 0.0; rz = 0.0; sa = False; sb = 0; ss = 0.0; blk = 0; rs = 0.0; tight = False; eq = np.zeros(n)
    for i in range(1, n):
        if gb != 0 and gb == vpos[i - 1] and vxp[i] > 0: gb = 0
        if real != 0:
            xp = -1.0
            if sa and real == 1 and l[i] <= ss: xp = min(o[i], ss) - tick; sa = False; blk = sb + H
            elif (not sa) and tight and real == gb and ((real == 1 and l[i] <= rs) or (real == -1 and h[i] >= rs)): xp = (min(o[i], rs) - tick) if real == 1 else (max(o[i], rs) + tick); gb = 0
            elif (not sa) and real == vpos[i - 1] and vxp[i] > 0: xp = vxp[i]
            if xp > 0: rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0; tight = False
        v = vpos[i]
        if vnew[i]: gb = v
        elif gb != v: gb = 0
        if sa and i - sb >= H: sa = False
        if (not sa) and evt[i] and i >= blk and gb == 0 and real == 0: sa = True; sb = i
        tg = 1 if sa else gb
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0; tight = False
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if sa: ss = ep * (1.0 - stp)
                elif vnew[i] and OPP[i]: tight = True; rs = ep * (1.0 - real * capO)
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq
rows = []
for tf, thr, cap in itertools.product(('1s', '4s'), (20, 25), (0.03, 0.05, 0.07)):
    a, p_, m_ = PX[tf][:3]; up = p_ > m_; OPP = np.zeros(n, np.bool_)
    sel = vnew & (vpos != 0); OPP[sel] = np.nan_to_num((a[sel] >= thr) & (((vpos[sel] < 0) & up[sel]) | ((vpos[sel] > 0) & ~up[sel])), nan=False)
    e = f5(o, h, l, c, TICK, vpos, vxp, vnew, OPP, cap, EVT, 1440, 0.08); r = ev(e, 0, 0, 0, v=f'F5 trende ters girişte zarar tavanı %{cap * 100:g} ({tf} ADX≥{thr})'); rows.append(r)
e0 = f5(o, h, l, c, TICK, vpos, vxp, vnew, np.zeros(n, np.bool_), 0.1, EVT, 1440, 0.08); B5 = ev(e0, 0, 0, 0)
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - B5[k]
print(f"\nF5 taban {B5['net']:.1f}"); print(R.sort_values('dnet', ascending=False)[['v', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']].round(1).to_string(index=False))
T = pd.read_excel('/home/claude/tvx/v30.xlsx', 'İşlemler'); g = T[T.Tip.str.contains('giriş')]
L24 = g[g.Tip.str.contains('Uzun') & g['Süre (çubuk)'].between(1435, 1442)]; print(f"\nTV V3.0: 24 s süren long (yan motor izi) {len(L24)} adet, net {L24['Net K/Z USDT'].sum():.1f} $ | %8 civarı stoplu long {((g.Tip.str.contains('Uzun')) & (g['Getiri %'].between(-8.6, -7.6))).sum()}")
R.to_csv('/home/claude/zecwork/adxfilt2.csv', index=False); pd.DataFrame(out).to_csv('/home/claude/zecwork/trende_ters_karne.csv', index=False)
