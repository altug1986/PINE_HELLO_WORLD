import numpy as np, pandas as pd
from eng import load, go, P, TICK
from lib import g_osc, zsc, rsi, g_btc
from ent import g_ent, feats, sig_of
from zz import g_rev
from sysB import mot2
# 14 BİLEŞEN SİSTEM — her biri 5 dk kapanışta −1 / 0 / +1 pozisyon
NAMES = ['1 DÖNÜŞ 24s (V1.0)', '2 DÖNÜŞ z 4s', '3 DÖNÜŞ z 8s', '4 DÖNÜŞ 4s uzama', '5 DÖNÜŞ 3g uzama', '6 DÖNÜŞ RSI14',
         '7 DÖNÜŞ likidasyon (OI)', '8 DÖNÜŞ OI kurulum', '9 DÖNÜŞ hacim klimaks', '10 DÖNÜŞ BTC göreli',
         '11 TAKİP BTC lider', '12 TAKİP %3', '13 TAKİP %8', '14 TAKİP günlük momentum 30g']
def fade(c, H, a, hold, sl):
    dP = (c / np.r_[np.full(H, np.nan), c[:-H]] - 1) * 100
    return g_ent(c, np.nan_to_num(np.where(dP <= -a, 1, np.where(dP >= a, -1, 0))).astype(np.int64), hold, sl)
def daily_mom(D, N=30, x=5.0):
    t = D['t']; k = t // 86400000; s = pd.Series(D['c']).groupby(k).last()
    m = (s / s.shift(N) - 1) * 100; sig = np.where(m >= x, 1, np.where(m <= -x, -1, 0))
    prev = pd.Series(np.r_[0, sig[:-1]], index=s.index)               # önceki tamamlanmış gün
    v = prev.reindex(k).to_numpy(); v = pd.Series(np.where(v == 0, np.nan, v)).ffill().fillna(0).to_numpy()
    return v.astype(np.int64)
def build(D, F=None):
    c = D['c']; F = feats(D) if F is None else F
    X = [mot2(D, F, dict(fam='PXA', H=288, aL=3.5, aS=5.5, hold=648, sl=3)),
         g_osc(c, zsc(c, 48), -3.5, 3.5, 0.5, -0.5, 8, 288),
         g_osc(c, zsc(c, 96), -3.5, 3.5, 0.5, -0.5, 8, 288),
         fade(c, 48, 6.0, 48, 10.0),
         fade(c, 864, 12.0, 1728, 10.0),
         g_osc(c, rsi(c, 14), 20, 80, 50, 50, 3, 144),
         go(D, P(liqPx=1.5, liqOi=1.5, liqExit=60, liqSL=5, liqBars=144), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int64))['tgt'],
         g_ent(c, sig_of(F, 'OI-KURULUM', 288, 5.0, 5.0, -1), 288, 10.0),
         g_ent(c, sig_of(F, 'HACİM-KLİMAKS', 288, 5.0, 1.3, -1), 288, 10.0),
         g_ent(c, sig_of(F, 'BTC-GÖRELİ', 288, 6.0, 0, -1), 288, 10.0),
         g_btc(c, D['xc'], 3, 2.0, 24, 1),
         g_rev(c, 3.0, 3.0), g_rev(c, 8.0, 8.0), daily_mom(D)]
    return np.stack(X).astype(np.int64)
if __name__ == '__main__':
    from lib import simR
    from met import SPLIT
    D = load(5); X = build(D); t = D['t']
    for nm, s in zip(NAMES, X):
        R = simR(D, s); p = R['pnl']; xt = t[R['xb']]
        print(f'{nm:30s} işlem {len(p):5d} net {p.sum():7.0f} PF {p[p>0].sum()/max(-p[p<0].sum(),1e-9):.2f} | ilk {p[xt<SPLIT].sum():6.0f} ikinci {p[xt>=SPLIT].sum():6.0f} | piyasada %{(s!=0).mean()*100:.0f}')
    np.save('comps.npy', X)
    print('bileşen korelasyonu (pozisyon):'); print(pd.DataFrame(np.corrcoef(X), index=[n.split()[0] for n in NAMES], columns=[n.split()[0] for n in NAMES]).round(2).to_string())
