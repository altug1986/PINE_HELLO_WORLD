import itertools, pickle, time, numpy as np, pandas as pd
from p1m import build, run, report, trades, SPL
from p1m2 import run2
from eng import rma, trr
D = build(5); c = D['c']; t = D['t']; tr = trr(D['h'], D['l'], c, True); atr = rma(tr, 14); r5 = np.r_[0, c[1:] / c[:-1] - 1]
m1 = pickle.load(open('D1.pkl', 'rb')); c1 = m1['c']; ret1 = np.r_[0, c1[1:] / c1[:-1] - 1]; a1 = pd.Series(np.abs(ret1)).rolling(60).mean().to_numpy()
j = np.searchsorted(m1['t'], t + 300000) - 1
DET = {f'5dk TR/ATR≥{k}': (np.nan_to_num(tr / np.r_[np.nan, atr[:-1]]) >= k, np.sign(r5)) for k in (2, 3, 5)}
DET.update({f'1dk/60dk≥{k}': (np.nan_to_num(np.abs(ret1[j]) / a1[j]) >= k, np.sign(ret1[j])) for k in (4, 6, 10)})
def fast(out, lab, **kw):
    R, _ = trades(D, out); p = R['pnl']; et = t[R['eb']]; xt = t[R['xb']]
    return dict(t=lab, net=round(p.sum()), n=len(p), PF=round(p[p > 0].sum() / -p[p < 0].sum(), 3), IS=round(p[et < SPL].sum()), OOS=round(p[et >= SPL].sum()), L200=round(p[xt >= t[-1] - 200 * 86400000].sum()), **kw)
b0 = fast(run(D), 'V6.6'); b1 = fast(run2(D), 'core2 regresyon'); print(b0, '\n', b1, flush=True)
rows = []; t0 = time.time()
for (dn, (S, sd)), e in itertools.product(DET.items(), (0.05, 0.1, 0.2)):
    E = e * S; up = E * (sd > 0); dn_ = E * (sd < 0)
    SET = [('A simetrik giriş+çıkış', dict(RL=E, RS=E, XL=E, XS=E)), ('C simetrik çıkış', dict(XL=E, XS=E)), ('E patlamaya ters pozisyondan çık', dict(XL=dn_, XS=up)),
           ('F patlama yönüne gir + ters pozisyondan çık', dict(RL=up, RS=dn_, XL=dn_, XS=up))]
    for g, (gF, gW) in [('tüm', (1.0, 1.0)), ('dönüş', (1.0, 0.0)), ('takip', (0.0, 1.0))]:
        SET += [(f'B simetrik giriş [{g}]', dict(RL=E, RS=E, gF=gF, gW=gW)), (f'D patlama yönüne giriş [{g}]', dict(RL=up, RS=dn_, gF=gF, gW=gW)), (f'G patlamaya ters giriş [{g}]', dict(RL=dn_, RS=up, gF=gF, gW=gW))]
    for lab, kw in SET: rows.append(fast(run2(D, **kw), lab, det=dn, eps=e))
r = pd.DataFrame(rows); r.to_pickle('milim.pkl'); print(len(r), f'{time.time()-t0:.0f}s')
pd.set_option('display.width', 250)
g = r.groupby('t').agg(test=('net', 'size'), ort=('net', 'mean'), max=('net', 'max'), IS_ort=('IS', 'mean'), OOS_ort=('OOS', 'mean'), baz_ustu=('net', lambda v: int((v > b0['net']).sum())), y6=('net', lambda v: int((v > b0['net'] * 1.06).sum())))
print(g.round(0).sort_values('ort', ascending=False).to_string())
print(r.sort_values('IS', ascending=False).head(10).to_string(index=False))
