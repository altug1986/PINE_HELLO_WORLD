# BAT V7.2 bileşimi (v72.py ile aynı): p1m çekirdek + yan motorlar (kapitülasyon > BTC OI boşalması > aylık z kapitülasyon > BTC çöküşü) + REV-25 trend katmanı
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); import p1m, rev25
from numba import njit
LAB = sys.argv[1]
@njit(cache=True)
def ent(s, c, hold, sl):
    m = len(s); out = np.zeros(m, np.int64); v = 0; e = 0.0; vb = 0
    for i in range(m):
        if v != 0 and ((c[i] / e - 1) * v <= -sl / 100 or i - vb >= hold): v = 0
        if s[i] != 0:
            if v != s[i]: v = s[i]; e = c[i]; vb = i
            else: vb = i
        out[i] = v
    return out
D = p1m.build(5); c = D['c']; t = D['t']; R_ = p1m.raw(); oT, oV = R_['oT'], R_['oV']; f5 = t // 300000 * 300000
j = np.searchsorted(oT, f5 - 300000, side='right') - 1; oi = np.where(j >= 0, oV[np.clip(j, 0, None)], np.nan); oi[f5 - 300000 < oT[0]] = np.nan
L = lambda a, k: np.r_[np.full(k, np.nan), a[:-k]]; ch = lambda a, k: (a / L(a, k) - 1) * 100
x = R_['x']; kx = (x[:, 0] // 300000).astype(np.int64); bc = pd.Series(x[:, 4], index=kx).groupby(level=0).last().reindex(t // 300000).ffill().to_numpy()
b24 = np.nan_to_num(ch(bc, 288)); p4 = np.nan_to_num(ch(c, 48)); o4 = np.nan_to_num(ch(oi, 48)); b1 = np.nan_to_num(ch(bc, 12))
X = pd.read_csv('btcoi5m.csv.gz', usecols=['source_create_time_utc', 'open_interest_qty']); xt = pd.to_datetime(X.source_create_time_utc).dt.tz_localize(None).to_numpy().astype('datetime64[ms]').astype(np.int64)
xv = X.open_interest_qty.to_numpy(float); m = xv > 0; xt, xv = xt[m], xv[m]; o_ = np.argsort(xt); xt, xv = xt[o_], xv[o_]
jx = np.searchsorted(xt, f5 - 300000, side='right') - 1; boi = np.where(jx >= 0, xv[np.clip(jx, 0, None)], np.nan); boi[f5 - 300000 < xt[0]] = np.nan
bo48, bo288, bp48 = ch(boi, 48), ch(boi, 288), ch(bc, 48); zs = lambda v: v / pd.Series(v).rolling(8640, min_periods=2000).std().to_numpy()
Z = dict(bo48=zs(bo48), bp48=zs(bp48), ao48=zs(ch(oi, 48)), ap48=zs(ch(c, 48)))
def build_out(whale):
    Fr = rev25.feats(D, whale == 'tv'); deep = Fr['deep']
    capB = ent(np.where(deep == -1, 0, (o4 <= -5) & (p4 <= -4)).astype(np.int64), c, 144, 5.0)
    crash = ent(np.where(deep == -1, 0, (b24 <= -10) & (b1 >= 0)).astype(np.int64), c, 288, 3.0)
    s = ((Z['bo48'] <= -2.5) & (Z['bp48'] <= -2.5)) | ((Z['ao48'] <= -2.5) & (Z['ap48'] <= -2.5)); capZ = ent(np.where(deep == -1, 0, np.nan_to_num(s)).astype(np.int64), c, 288, 5.0)
    flush = ent(np.where(deep == -1, 0, np.nan_to_num(bo288) <= -10).astype(np.int64), c, 288, 3.0)
    ov = crash.copy()
    for a_ in reversed([capB, flush, capZ]): ov = np.where(a_ != 0, a_, ov)
    return rev25.apply25(D, p1m.run(D, whale=whale, ovr=ov), Fr)
rows = []; MID = (pd.Timestamp('2023-12-11') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
for wh in ('tam', 'tv', 'yok'):
    out = build_out(wh); r = p1m.report(D, out, f'{LAB} · BAT V7.2 · balina {wh}'); R, op = p1m.trades(D, out); p = R['pnl']; et = t[R['eb']]
    yr = pd.Series(p, index=pd.to_datetime(t[R['xb']], unit='ms').year).groupby(level=0).sum()
    r.update(ilkYarı=round(p[et < MID].sum()), ikinciYarı=round(p[et >= MID].sum()), ralliHariç=round(p.sum() - p[(et >= 1756684800000) & (et < 1767225600000)].sum()), yıllar=' '.join(f'{v:.0f}' for v in yr.values))
    rows.append(r); np.save(f'/home/claude/zbat/out_{LAB}_{wh}.npy', out)
pd.set_option('display.width', 320); print(pd.DataFrame(rows).to_string(index=False))
