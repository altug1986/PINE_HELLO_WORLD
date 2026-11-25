"""ZEC 2M — SHORT GİRİŞ TARAMASI (organize, olay bazlı).

Her tetik yalnız SHORT. Olay = tetik barı, önceki 30 barda olay yoksa (bağımsız). Bar kapanışında giriş.
Ölçü (bps, short yönünde, maliyet 8,3 bps düşülmüş): h15 / h60 / h240 ileri getiri; MAE P90 (60 bar içinde en kötü bar içi yükseliş).
Bölünme: IS (ilk yarı) / OOS (ikinci yarı), yıl tutarlılığı (h60 net > 0 olan yıl sayısı / 6), rejim (4 sa ADX).
Referans: aynı rejimde koşulsuz short (drift) — tetiğin bunu aşması gerekir.
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
from zec_data import load
import z01, z02

COST = 8.3
B = load(2, src='/home/claude/master/'); F = z01.prep(B)
c, h, l, a, v = F['c'], F['h'], F['l'], F['atr'], F['v']; n = len(c)
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64()); yr = B.index.year.values
A4 = np.load('/home/claude/zecwork/adx_4h.npy'); RG = np.where(A4[:, 0] >= 20, A4[:, 1], 0)       # +1 UP, −1 DN, 0 YATAY
sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]
roll = lambda x, w, f: getattr(pd.Series(x).rolling(w, min_periods=max(2, w // 2)), f)().values
FW = {H: (1 - np.r_[c[H:] / c[:-H], np.full(H, np.nan)]) * 1e4 - COST for H in (15, 60, 240)}          # short getirisi
fmax60 = pd.Series(h[::-1]).rolling(60, min_periods=1).max().values[::-1]; fmax60 = np.r_[fmax60[1:], np.nan]
MAE60 = (fmax60 / c - 1) * 100

T = {}
# A — fiyat yapısı
for N in (30, 90, 360):
    LL = roll(l, N, 'min'); LL = sh(LL, 1); T[f'A1 Donchian{N} kırılım'] = (c < LL) & (sh(c, 1) >= sh(LL, 1))
for N in (60, 240):
    HH = sh(roll(h, N, 'max'), 1); brk = roll((h > HH).astype(float), 5, 'max') > 0
    T[f'A2 Başarısız kırılım{N}'] = brk & (c < sh(HH, 5) - a) & (c < sh(c, 1))
m20 = (c - sh(c, 20)) / a
for k in (6, 8, 10, 12):
    T[f'A3 Pump fade 20bar≥{k}ATR'] = (m20 >= k) & (sh(m20, 1) < k)
pmp = roll((m20 >= 8).astype(float), 240, 'max') > 0
for W in (15, 30):
    LLw = sh(roll(l, W, 'min'), 1); T[f'A4 Pump sonrası dip kırılımı{W}'] = pmp & (c < LLw) & (sh(c, 1) >= sh(LLw, 1))
d = np.diff(c, prepend=c[0]); g = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / 14, adjust=False).mean(); ls_ = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / 14, adjust=False).mean()
rsi = (100 - 100 / (1 + g / ls_)).values
for t_ in (75, 80, 85):
    T[f'A5 RSI>{t_} çıkış'] = (sh(rsi, 1) >= t_) & (rsi < t_)
mu = roll(c, 120, 'mean'); sd = pd.Series(c).rolling(120).std(ddof=0).values; up_ = mu + 2.5 * sd
T['A6 BB üst red'] = (h > up_) & (c < up_) & (c < sh(c, 1))
ze = (c - z02.ema(c, 240)) / a
for k in (6, 8, 10):
    T[f'A7 EMA240 sapma ≥{k} dönüş'] = (sh(ze, 1) >= k) & (ze < k)
e60 = z02.ema(c, 60); e240 = z02.ema(c, 240)
T['A8 Düşüşte EMA60 retest red'] = (c < e240) & (e60 < e240) & (h >= e60) & (c < e60) & (c < sh(c, 1))
# B — OI mikro
lo = np.log(B.oi_zec.values.astype(float))
def oiz(w):
    dd = lo - sh(lo, w); return dd / pd.Series(dd).rolling(2160, min_periods=720).std().values, dd
for w in (8, 15, 30):
    z_, _ = oiz(w); dp = (c - sh(c, w)) / a
    for zt in (1.5, 2.5):
        cond = (z_ >= zt) & (dp <= -1); T[f'B1 Yeni short parası OI{w} z≥{zt}'] = cond & ~(sh(cond.astype(float), 1) > 0)
z15, _ = oiz(15); HH60 = sh(roll(h, 60, 'max'), 1)
T['B2 Tepede OI şişmesi (sıkışan long)'] = (z15 >= 2) & (c >= HH60) & (c < sh(c, 1))
z8, _ = oiz(8); dp8 = (c - sh(c, 8)) / a
T['B3 Short sıkışması bitişi'] = (z8 <= -2) & (dp8 >= 1.5)
for N in (30, 60, 120):
    HHn = sh(roll(h, N, 'max'), 1); T[f'B4 OI uyumsuzluğu tepe{N}'] = (c >= HHn) & ((lo - sh(lo, N)) < -0.005)
# C — basis / funding
bs = B.basis_bps.values; bz = (bs - roll(bs, 720, 'mean')) / roll(bs, 720, 'std')
for k in (2, 3):
    T[f'C1 Basis primi z≥{k}'] = (bz >= k) & (sh(bz, 1) < k)
p95 = pd.Series(bs).rolling(21600, min_periods=5000).quantile(0.95).values
T['C2 Basis ≥ 30g P95'] = (bs >= p95) & (sh(bs, 1) < sh(p95, 1))
fu = B.funding_pct.values
for k in (0.03, 0.05, 0.1):
    T[f'C3 Funding ≥{k}%'] = (fu >= k) & (sh(fu, 1) < k)
# D — pozisyonlanma / balina
tp = np.log(B.tt_pos_ls.values)
for k in (0.03, 0.06):
    dtp = tp - sh(tp, 120); T[f'D1 Top trader long artışı ≥{int(k*100)}% (kontra)'] = (dtp >= k) & (sh(dtp, 1) < k)
gl = B.glb_ls.values; gz = (gl - roll(gl, 5040, 'mean')) / roll(gl, 5040, 'std'); T['D2 Genel L/S z≥2 (kontra)'] = (gz >= 2) & (sh(gz, 1) < 2)
qv = roll(B.quote_volume.values, 30, 'sum'); wn = roll(B.wh_sell_q50k.values - B.wh_buy_q50k.values, 30, 'sum') / qv
for k in (0.05, 0.10):
    T[f'D3 Balina net satış ≥{int(k*100)}% (takip)'] = (wn >= k) & (sh(wn, 1) < k)
T['D4 Tepede balina alımı (emilim)'] = (-wn >= 0.10) & (c >= HH60) & (c < sh(c, 1))
# E — akış / emir defteri
td = roll(B.taker_delta_base.values, 10, 'sum') / roll(v, 10, 'sum'); dp10 = (c - sh(c, 10)) / a
T['E1 Taker alım tükenişi'] = (td >= 0.3) & (dp10 < 0.5) & (sh(td, 1) < 0.3)
bi = ((B.book_m1 - B.book_p1) / (B.book_m1 + B.book_p1)).values
T['E2 Emir defteri satış ağır'] = (bi <= -0.3) & (sh(bi, 1) > -0.3)
# F — çapraz piyasa
bt = B.btc_close.values; xm = B.xmr_close.values
rb = (bt / sh(bt, 30) - 1) * 100; rz = (c / sh(c, 30) - 1) * 100
T['F1 BTC düştü ZEC gecikti'] = (rb <= -1) & (rz > -0.3) & (sh(rb, 1) > -1)
dz = np.r_[0, np.diff(np.log(c))]; db = np.r_[0, np.diff(np.log(bt))]
beta = np.nan_to_num(roll(dz * db, 720, 'mean') / roll(db * db, 720, 'mean')); res = roll(dz - beta * db, 30, 'sum')
rz_ = res / (roll(dz - beta * db, 720, 'std') * np.sqrt(30)); T['F2 Kendi pump (artık z≥2) fade'] = (rz_ >= 2) & (sh(rz_, 1) < 2)
rx = (xm / sh(xm, 30) - 1) * 100; T['F3 XMR düşüyor ZEC yükseliyor'] = (rx <= -1) & (rz >= 1)

def dedupe(m):
    i = np.where(np.nan_to_num(m).astype(bool))[0]; keep = []; last = -10**9
    for j in i:
        if j - last >= 30: keep.append(j); last = j
    return np.array(keep, dtype=int)

rows = []; base = {}
for lab, val in (('TÜM', None), ('UP', 1), ('DN', -1), ('YAT', 0)):
    m = np.ones(n, bool) if val is None else (RG == val)
    base[lab] = (np.nanmean(FW[60][:mid][m[:mid]]), np.nanmean(FW[60][mid:][m[mid:]]))
for name, trig in T.items():
    ev = dedupe(trig)
    for lab, val in (('TÜM', None), ('UP', 1), ('DN', -1), ('YAT', 0)):
        e = ev if val is None else ev[RG[ev] == val]
        if len(e) < 40: continue
        isk = e < mid; r = dict(tetik=name, rejim=lab, n=len(e), nIS=int(isk.sum()))
        for H in (15, 60, 240):
            x = FW[H][e]; r[f'h{H}'] = np.nanmean(x); r[f'h{H}IS'] = np.nanmean(x[isk]); r[f'h{H}OOS'] = np.nanmean(x[~isk])
        r['kaz%60'] = np.nanmean(FW[60][e] > 0) * 100; r['MAE90'] = np.nanpercentile(MAE60[e], 90)
        r['yıl+'] = sum(np.nanmean(FW[60][e[yr[e] == y]]) > 0 for y in range(2021, 2027) if (yr[e] == y).sum() >= 5)
        r['drift60'] = np.mean(base[lab]); rows.append(r)
D = pd.DataFrame(rows); D.to_csv('/home/claude/pine_hello_world/ZEC/research/SHORT_SCAN.csv', index=False)
pd.set_option('display.width', 250)
print('Koşulsuz short h60 (drift, net): ' + ' · '.join(f'{k} {a_:.1f}/{b_:.1f}' for k, (a_, b_) in base.items()))
cols = ['tetik', 'rejim', 'n', 'h15IS', 'h15OOS', 'h60IS', 'h60OOS', 'h240IS', 'h240OOS', 'kaz%60', 'MAE90', 'yıl+']
G = D[(D.h60IS > 0) & (D.h60OOS > 0) & (D['yıl+'] >= 4)].sort_values('h60', ascending=False)
print(f'\nİKİ YARIDA NET POZİTİF + ≥4/6 YIL (h60, maliyet sonrası): {len(G)} / {len(D)}'); print(G[cols].round(1).to_string(index=False))
print('\nEn büyük örneklemli 12 tetik (TÜM rejim):'); print(D[D.rejim == 'TÜM'].sort_values('n', ascending=False).head(12)[cols].round(1).to_string(index=False))
