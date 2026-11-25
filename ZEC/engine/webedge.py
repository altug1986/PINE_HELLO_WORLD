# WEB'DEN 3 YENİ VERİ AİLESİ — ön test (yön bilgisi var mı, iki yarıda tutuyor mu, maliyeti geçer mi)
# 1) Çeyrek saat (:00/:15/:30/:45) taker dengesizliği → 4–12 s devam (arXiv 2607.09426)  2) Emir defteri derinlik dengesizliği  3) Funding ödeme saati etkisi
import numpy as np, pandas as pd, glob
cols = ['time', 'close', 'volume', 'taker_buy_base', 'book_m1', 'book_p1', 'book_m5', 'book_p5', 'funding_pct', 'premium_bps']
M = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))], ignore_index=True)
Z = np.load('/home/claude/zecwork/base28.npz'); assert len(M) == len(Z['c']) and np.allclose(M.close.values, Z['c']); mid = int(Z['mid'])
c = M.close.values; t = M.time.values; n = len(c); mm = (t // 60000) % 60; hh = (t // 3600000) % 24
imb = np.where(M.volume.values > 0, 2 * M.taker_buy_base.values / np.maximum(M.volume.values, 1e-12) - 1, np.nan)
def fwd(H): r = np.full(n, np.nan); r[:-H] = c[H:] / c[:-H] - 1; return r
FW = {H: fwd(H) for H in (30, 240, 480, 720)}
def ic(x, y, idx):
    """Spearman IC, iki yarı ayrı + 20 zaman bloğu t (bloklar arası IC ortalaması / se)."""
    out = {}
    for nm, sel in (('1.yarı', idx[idx < mid]), ('2.yarı', idx[idx >= mid])):
        ok = sel[np.isfinite(x[sel]) & np.isfinite(y[sel])]
        if len(ok) < 200: out[nm] = (np.nan, np.nan, len(ok)); continue
        r = pd.Series(x[ok]).corr(pd.Series(y[ok]), method='spearman'); B = np.array_split(ok, 20)
        bi = np.array([pd.Series(x[b]).corr(pd.Series(y[b]), method='spearman') for b in B]); out[nm] = (r, bi.mean() / (bi.std() / np.sqrt(len(bi))), len(ok))
    return out
def qspread(x, y, idx):   # en üst %20 − en alt %20 sinyal grubunun ileri getiri farkı (bps)
    ok = idx[np.isfinite(x[idx]) & np.isfinite(y[idx])]; q1, q4 = np.quantile(x[ok], [0.2, 0.8]); return (y[ok][x[ok] >= q4].mean() - y[ok][x[ok] <= q1].mean()) * 1e4
rows = []
def add(lab, x, idx):
    for H in (30, 240, 480, 720):
        r = ic(x, FW[H], idx); rows.append(dict(sinyal=lab, H_dk=H, IC_1=r['1.yarı'][0], t_1=r['1.yarı'][1], IC_2=r['2.yarı'][0], t_2=r['2.yarı'][1], adet=r['1.yarı'][2] + r['2.yarı'][2], üst_alt_bps=qspread(x, FW[H], idx)))
q = np.flatnonzero(np.isin(mm, (15, 30, 45))); q0 = np.flatnonzero(mm == 0); ordn = np.flatnonzero(mm == 7)
add('çeyrek saat (:15/:30/:45) dengesizlik', imb, q); add('saat başı (:00) dengesizlik', imb, q0); add('sıradan dakika (:07) dengesizlik [kontrol]', imb, ordn)
cs = np.where(np.isin(mm, (15, 30, 45)), np.nan_to_num(imb), 0.0); cum = np.cumsum(cs)
for K in (16, 48):   # son K×15 dk içindeki çeyrek saat dengesizlik toplamı (4 s / 12 s)
    agg = np.full(n, np.nan); W = K * 15; agg[W:] = cum[W:] - cum[:-W]; add(f'çeyrek saat dengesizlik toplamı son {W // 60} s', agg, q)
tbv = np.nan_to_num(M.taker_buy_base.values); vv = M.volume.values; ct = np.cumsum(2 * tbv - vv); cv = np.cumsum(vv)
for W in (240, 720):   # tüm dakikaların net taker akışı (CVD) / hacim
    g = np.full(n, np.nan); g[W:] = (ct[W:] - ct[:-W]) / np.maximum(cv[W:] - cv[:-W], 1e-9); add(f'tüm dakikalar net taker akışı son {W // 60} s [kontrol]', g, q)
bm1, bp1, bm5, bp5 = (M[k].values for k in ('book_m1', 'book_p1', 'book_m5', 'book_p5'))
d1 = (bm1 - bp1) / (bm1 + bp1); d5 = (bm5 - bp5) / (bm5 + bp5); qq = np.flatnonzero((mm % 15 == 14) & np.isfinite(d5))
print(f'emir defteri verisi: {pd.to_datetime(t[qq[0]], unit="ms").date()} → {pd.to_datetime(t[qq[-1]], unit="ms").date()} ({len(qq)} çeyrek)')
add('emir defteri dengesizlik ±%1 (alış−satış)', d1, qq); add('emir defteri dengesizlik ±%5', d5, qq)
fs = np.flatnonzero((mm == 0) & (hh % 8 == 0)); fr = M.funding_pct.values
pre = np.full(n, np.nan); pre[60:] = c[59:-1] / c[:-60] - 1                  # ödemeden önceki 1 saat (T−60 → T−1 kapanış)
fpre = np.full(n, np.nan); fpre[1:] = fr[:-1]
add('funding oranı → ödeme SONRASI getiri', fpre, fs)
r1 = []
for nm, sel in (('1.yarı', fs[fs < mid]), ('2.yarı', fs[fs >= mid])):
    f_ = fpre[sel]; p_ = pre[sel]; ok = np.isfinite(f_) & np.isfinite(p_)
    r1.append(f"{nm}: funding>%0,03 iken ödeme öncesi 1 s ort {np.mean(p_[ok & (f_ > 0.03)]) * 1e4:+.1f} bps (n={np.sum(ok & (f_ > 0.03))}), funding<0 iken {np.mean(p_[ok & (f_ < 0)]) * 1e4:+.1f} bps (n={np.sum(ok & (f_ < 0))}), hepsi {np.mean(p_[ok]) * 1e4:+.1f}")
R = pd.DataFrame(rows); pd.set_option('display.width', 250); print(R.round(4).to_string(index=False)); print('\n'.join(r1))
R.to_csv('/home/claude/zecwork/webedge.csv', index=False)
