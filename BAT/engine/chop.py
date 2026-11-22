import numpy as np, pandas as pd, itertools
from vote import vote_pos
from eng import load
from lib import simR, cut
from met import metrics, SPLIT
# TESTERE / YATAY TRAŞLAMA: fiyat VHF, OI VHF (dikey-yatay filtre), BTC ve BAT çoklu zaman dilimi hareket büyüklüğü
D0 = load(5); X0 = np.load('comps.npy'); c = D0['c']; xc = D0['xc']; oi = pd.Series(np.r_[np.nan, D0['oi'][:-1]]).ffill().to_numpy()
W6 = np.array([3, 3, 0.5, 0, 0, 1, 3, 3, 0.5, 1, 0.5, 0.5, 2, 0], float); DON = np.array([1] * 10 + [0] * 4, bool)
def vhf(a, n):
    s = pd.Series(a); return ((s.rolling(n).max() - s.rolling(n).min()) / s.diff().abs().rolling(n).sum()).to_numpy()
def ret(a, k): return np.abs(np.r_[np.full(k, np.nan), a[k:] / a[:-k] - 1] * 100)
IND = {'fiyat VHF 4s': vhf(c, 48), 'fiyat VHF 24s': vhf(c, 288), 'OI VHF 4s': vhf(oi, 48), 'OI VHF 24s': vhf(oi, 288),
       'BTC |4s|': ret(xc, 48), 'BTC |24s|': ret(xc, 288), 'BAT |4s|': ret(c, 48), 'BAT |24s|': ret(c, 288)}
k0 = np.searchsorted(D0['t'], SPLIT); MID = np.int64(pd.Timestamp('2022-06-15').value // 10**6); km = np.searchsorted(D0['t'], MID)
DI = cut(D0, SPLIT); XI = X0[:, :k0]
def halves(tg):
    out = []
    for a, b in [(0, km), (km, k0)]:
        Dh = {k: (v[a:b] if isinstance(v, np.ndarray) and len(v) == len(DI['t']) else v) for k, v in DI.items()}
        m = metrics(Dh, simR(Dh, tg[a:b])); out.append(m['net'] - 1.5 * m['p98'])
    return min(out), out
tg0, _ = vote_pos(XI, W6, 0.05, 0.0, 1); b0 = halves(tg0); print('V5.0 ilk dönem yarı skorları', [round(x) for x in b0[1]])
tgM, _ = vote_pos(XI, W6 * DON, 0.05, 0.0, 1); tgT, _ = vote_pos(XI, W6 * np.where(DON, 0.5, 2.0), 0.05, 0.0, 1)
rows = []
for nm, x in IND.items():
    xi = x[:k0]
    for q in [0.2, 0.33, 0.5]:
        th = np.nanquantile(xi, q); chop = np.nan_to_num(xi, nan=th + 1) < th          # testere = düşük VHF / küçük hareket
        for act in ['testerede FLAT', 'testerede sadece dönüş', 'testerede dönüş + trendde takip ağır']:
            if act == 'testerede FLAT': tg = np.where(chop, 0, tg0)
            elif act == 'testerede sadece dönüş': tg = np.where(chop, tgM, tg0)
            else: tg = np.where(chop, tgM, tgT)
            s, hs = halves(tg.astype(np.int64)); rows.append(dict(gösterge=nm, eşik_q=q, eylem=act, sc=s, a=hs[0], b=hs[1], testere_payı=round(chop.mean(), 2)))
df = pd.DataFrame(rows); pd.set_option('display.width', 250)
print('varyant', len(df), '| iki yarıda da V5.0 üstü:', int(((df.a > b0[1][0]) & (df.b > b0[1][1])).sum()))
print(df.sort_values('sc', ascending=False).head(8).round(0).to_string(index=False))
