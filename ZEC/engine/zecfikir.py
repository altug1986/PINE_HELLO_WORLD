# ZEC'E ÖZEL FİKİR BLOĞU (web: ZEC vadeli ağırlıklı — vadeli hacmi ~10× spot, short sıkışmaları, XMR ile gizlilik rotasyonu)
# A) vadeli mi spot mu sürüklüyor (baz farkı değişimi)  B) sıkışma yakıtı (funding/OI/L-S kalabalık)  C) ZEC–XMR rotasyonu  D) spekülasyon yoğunluğu (hacim/OI)  E) likidasyon tipi olaylar
import numpy as np, pandas as pd, glob
cols = ['time', 'close', 'quote_volume', 'basis_bps', 'premium_bps', 'funding_pct', 'oi_usdt', 'tt_pos_ls', 'glb_ls', 'xmr_close', 'btc_close']
M = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))], ignore_index=True)
Z = np.load('/home/claude/zecwork/base28.npz'); mid = int(Z['mid']); c = M.close.values; n = len(c); S = pd.Series
def ff(x): return S(x).ffill().values
def lagd(x, k): return x - np.r_[np.full(k, np.nan), x[:-k]]
def zs(x, W): s = S(x); return ((s - s.rolling(W, min_periods=W // 4).mean()) / s.rolling(W, min_periods=W // 4).std()).values
def fwd(H): r = np.full(n, np.nan); r[:-H] = c[H:] / c[:-H] - 1; return r
FW = {H: fwd(H) for H in (240, 720, 1440)}; lc = np.log(c); idx = np.arange(1440 * 30, n - 1441, 60)
bas = ff(M.basis_bps.values); fund = ff(M.funding_pct.values); oi = ff(M.oi_usdt.values); xmr = ff(M.xmr_close.values); btc = ff(M.btc_close.values)
p4 = lagd(lc, 240); p24 = lagd(lc, 1440); db4 = lagd(bas, 240)
SIG = {}
SIG['A1 spot öncülüğü: 4 s hareket yönünde baz farkı DÜŞÜŞÜ (spot sürüklüyor)'] = -np.sign(p4) * db4
SIG['A2 momentum 4 s (karşılaştırma)'] = p4
SIG['A3 baz farkı seviyesi (vadeli pahalı → +)'] = bas
SIG['B1 funding z (7 g) — kalabalık long +'] = zs(fund, 10080)
SIG['B2 genel L/S oranı log z (7 g)'] = zs(np.log(np.maximum(ff(M.glb_ls.values), 1e-6)), 10080)
SIG['B3 top trader pozisyon L/S log z (7 g)'] = zs(np.log(np.maximum(ff(M.tt_pos_ls.values), 1e-6)), 10080)
SIG['B4 OI 24 s değişimi'] = lagd(np.log(oi), 1440)
SIG['C1 ZEC/XMR göreli 24 s'] = p24 - lagd(np.log(xmr), 1440)
SIG['C2 ZEC/XMR göreli 3 g'] = lagd(lc, 4320) - lagd(np.log(xmr), 4320)
SIG['C3 XMR 24 s getirisi (öncü mü?)'] = lagd(np.log(xmr), 1440)
qv = S(M.quote_volume.values).rolling(1440, min_periods=720).sum().values
SIG['D1 spekülasyon: 24 s hacim / OI (z, 30 g)'] = zs(np.log(np.maximum(qv / np.maximum(oi, 1), 1e-9)), 43200)
SIG['D2 hacim/OI z × 4 s yön'] = SIG['D1 spekülasyon: 24 s hacim / OI (z, 30 g)'] * np.sign(p4)
def ic(x, y, sel):
    ok = sel[np.isfinite(x[sel]) & np.isfinite(y[sel])]
    if len(ok) < 300: return np.nan, np.nan
    r = S(x[ok]).corr(S(y[ok]), method='spearman'); B = np.array_split(ok, 20); bi = np.array([S(x[b]).corr(S(y[b]), method='spearman') for b in B]); return r, bi.mean() / (bi.std() / np.sqrt(20))
rows = []
for nm, x in SIG.items():
    r = dict(sinyal=nm)
    for H in (240, 720, 1440):
        a, ta = ic(x, FW[H], idx[idx < mid]); b, tb = ic(x, FW[H], idx[idx >= mid]); r[f'{H // 60}s IC 1/2'] = f'{a:+.3f}/{b:+.3f}'; r[f'{H // 60}s t 1/2'] = f'{ta:+.1f}/{tb:+.1f}'
    rows.append(r)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70); print(pd.DataFrame(rows).to_string(index=False))
# OLAYLAR (240 bar arayla tekil), sonraki 4/12/24 s getiri (bps) — olay yönünde
oi1 = lagd(np.log(oi), 60); p1 = lagd(lc, 60); zp1 = zs(p1, 43200)
EVT = {'E1 long tasfiyesi: 1 s fiyat z ≤ −3 ve OI 1 s ≤ −%3 → long (dönüş)': ((zp1 <= -3) & (oi1 <= -0.03), 1),
       'E2 short sıkışması: 1 s fiyat z ≥ +3 ve OI 1 s ≤ −%3 → long (devam)': ((zp1 >= 3) & (oi1 <= -0.03), 1),
       'E3 short sıkışması → short (dönüş)': ((zp1 >= 3) & (oi1 <= -0.03), -1),
       'E4 yeni short yığılması: fiyat 4 s ≥ +%5, funding < 0, OI 24 s ≥ +%10 → long (sıkışma yakıtı)': ((p4 >= 0.05) & (fund < 0) & (lagd(np.log(oi), 1440) >= 0.10), 1),
       'E5 kalabalık long: fiyat 4 s ≤ −%5, funding > %0,02, OI 24 s ≥ +%10 → short (long tasfiye devamı)': ((p4 <= -0.05) & (fund > 0.02) & (lagd(np.log(oi), 1440) >= 0.10), -1),
       'E6 vadeli öncülüklü yükseliş: 4 s ≥ +%5 ve baz farkı 4 s ≥ +15 bps → short (dönüş)': ((p4 >= 0.05) & (db4 >= 15), -1),
       'E7 spot öncülüklü yükseliş: 4 s ≥ +%5 ve baz farkı 4 s ≤ 0 → long (devam)': ((p4 >= 0.05) & (db4 <= 0), 1),
       'E8 spot öncülüklü düşüş: 4 s ≤ −%5 ve baz farkı 4 s ≥ 0 → short (devam)': ((p4 <= -0.05) & (db4 >= 0), -1)}
er = []
for nm, (m, d) in EVT.items():
    st = np.flatnonzero(np.nan_to_num(m, nan=0) > 0); k = []; last = -10 ** 9
    for s_ in st:
        if s_ - last >= 240 and s_ < n - 1441: k.append(s_); last = s_
    k = np.array(k, dtype=np.int64); r = dict(olay=nm, adet=len(k))
    for H in (240, 720, 1440):
        if len(k) == 0: continue
        f = FW[H][k] * d * 1e4; r[f'{H // 60}s bps 1/2'] = f'{f[k < mid].mean() if (k < mid).any() else np.nan:+.0f}/{f[k >= mid].mean() if (k >= mid).any() else np.nan:+.0f}'; r[f'{H // 60}s isabet'] = f'{(f > 0).mean() * 100:.0f}%'
    r['adet 1/2'] = f'{(k < mid).sum()}/{(k >= mid).sum()}'; er.append(r)
print(); print(pd.DataFrame(er).to_string(index=False))
pd.DataFrame(rows).to_csv('/home/claude/zecwork/zecfikir_ic.csv', index=False); pd.DataFrame(er).to_csv('/home/claude/zecwork/zecfikir_olay.csv', index=False)
