# TradingView "Technical Ratings" klonu (açık kaynak göstergenin mantığı): 15 hareketli ortalama oyu + 11 osilatör oyu → (MA + OSC)/2
# > 0,5 Güçlü Al · > 0,1 Al · −0,1…0,1 Nötr · < −0,1 Sat · < −0,5 Güçlü Sat
import numpy as np, pandas as pd
def _rma(x, n): return x.ewm(alpha=1 / n, adjust=False).mean()
def _rsi(c, n): d = c.diff(); return 100 - 100 / (1 + _rma(d.clip(lower=0), n) / _rma((-d).clip(lower=0), n))
def _sig(buy, sell, valid): r = np.where(buy, 1.0, np.where(sell, -1.0, 0.0)); return pd.Series(np.where(valid, r, np.nan), buy.index)
def _hma(c, n): w = lambda s, k: s.rolling(k).apply(lambda a: np.dot(a, np.arange(1, k + 1)) / np.arange(1, k + 1).sum(), raw=True); return w(2 * w(c, n // 2) - w(c, n), int(np.sqrt(n)))
def rating(D):
    o, h, l, c, v = D.o, D.h, D.l, D.c, D.v; ma = []
    for n in (10, 20, 30, 50, 100, 200):
        for m in (c.rolling(n).mean(), c.ewm(span=n, adjust=False).mean()): ma.append(_sig(c > m, c < m, m.notna()))
    dc = lambda n: (h.rolling(n).max() + l.rolling(n).min()) / 2; cl, bl, l2 = dc(9), dc(26), dc(52); l1 = (cl + bl) / 2
    ma.append(_sig((l1 > l2) & (c > l1) & (c < bl) & (c.shift() < cl) & (c > cl), (l2 > l1) & (c < l2) & (c > bl) & (c.shift() > cl) & (c < cl), l2.notna()))
    vw = (c * v).rolling(20).sum() / v.rolling(20).sum(); ma.append(_sig(c > vw, c < vw, vw.notna())); hm = _hma(c, 9); ma.append(_sig(c > hm, c < hm, hm.notna()))
    MA = pd.concat(ma, axis=1).mean(axis=1); osc = []
    r = _rsi(c, 14); osc.append(_sig((r < 30) & (r.shift() < r), (r > 70) & (r.shift() > r), r.notna()))
    st = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min()); k = st.rolling(3).mean(); d = k.rolling(3).mean()
    osc.append(_sig((k < 20) & (d < 20) & (k > d), (k > 80) & (d > 80) & (k < d), d.notna()))
    tp = (h + l + c) / 3; md = tp.rolling(20).apply(lambda a: np.abs(a - a.mean()).mean(), raw=True); cci = (tp - tp.rolling(20).mean()) / (0.015 * md)
    osc.append(_sig((cci < -100) & (cci > cci.shift()), (cci > 100) & (cci < cci.shift()), cci.notna()))
    pc = c.shift(); tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1); up = h.diff(); dn = -l.diff(); at = _rma(tr, 14)
    pdi = 100 * _rma(pd.Series(np.where((up > dn) & (up > 0), up, 0.0), c.index), 14) / at; mdi = 100 * _rma(pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), c.index), 14) / at
    adx = _rma(100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan), 14)
    osc.append(_sig((adx > 20) & (pdi.shift() < mdi.shift()) & (pdi > mdi), (adx > 20) & (pdi.shift() > mdi.shift()) & (pdi < mdi), adx.notna()))
    hl2 = (h + l) / 2; ao = hl2.rolling(5).mean() - hl2.rolling(34).mean()
    osc.append(_sig(((ao > 0) & (ao.shift() < 0)) | ((ao > 0) & (ao.shift() > 0) & (ao > ao.shift()) & (ao.shift(2) > ao.shift())), ((ao < 0) & (ao.shift() > 0)) | ((ao < 0) & (ao.shift() < 0) & (ao < ao.shift()) & (ao.shift(2) < ao.shift())), ao.notna()))
    mo = c - c.shift(10); osc.append(_sig(mo > mo.shift(), mo < mo.shift(), mo.notna()))
    mc = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean(); sg = mc.ewm(span=9, adjust=False).mean(); osc.append(_sig(mc > sg, mc < sg, sg.notna()))
    sr = 100 * (r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min()); sk = sr.rolling(3).mean(); sd = sk.rolling(3).mean(); m50 = c.rolling(50).mean()
    osc.append(_sig((c < m50) & (sk < 20) & (sd < 20) & (sk > sd), (c > m50) & (sk > 80) & (sd > 80) & (sk < sd), sd.notna()))
    wr = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min()); osc.append(_sig((wr < -80) & (wr > wr.shift()), (wr > -20) & (wr < wr.shift()), wr.notna()))
    e13 = c.ewm(span=13, adjust=False).mean(); bu = h - e13; be = l - e13
    osc.append(_sig((c > m50) & (be < 0) & (be > be.shift()), (c < m50) & (bu > 0) & (bu < bu.shift()), m50.notna()))
    bp = c - pd.concat([l, pc], axis=1).min(axis=1); trr = pd.concat([h, pc], axis=1).max(axis=1) - pd.concat([l, pc], axis=1).min(axis=1)
    av = lambda n: bp.rolling(n).sum() / trr.rolling(n).sum(); uo = 100 * (4 * av(7) + 2 * av(14) + av(28)) / 7; osc.append(_sig(uo > 70, uo < 30, uo.notna()))
    OSC = pd.concat(osc, axis=1).mean(axis=1); return MA, OSC, (MA + OSC) / 2
def etiket(x): return 'Güçlü Al' if x > 0.5 else 'Al' if x > 0.1 else 'Nötr' if x >= -0.1 else 'Sat' if x >= -0.5 else 'Güçlü Sat'
