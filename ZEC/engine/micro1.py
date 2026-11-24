import sys, json, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01
B = load(2, src='/home/claude/master/'); F = z01.prep(B); c = F['c']; a = F['atr']; n = len(c)
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64()); yr = B.index.year.values
A4 = np.load('/home/claude/zecwork/adx_4h.npy'); R = np.where(A4[:, 0] >= 20, A4[:, 1], np.where(A4[:, 0] < 20, 0, 9))
fw = {h: np.r_[c[h:] / c[:-h] - 1, np.full(h, np.nan)] * 1e4 for h in (15, 60, 240)}
oi = B.oi_zec.values.astype(float); lo = np.log(oi)
def sh(x, k): return np.r_[np.full(k, np.nan), x[:-k]]
sig = {}
for w in (3, 8, 15, 30):
    doi = lo - sh(lo, w); sd = pd.Series(doi).rolling(720 * 3, min_periods=720).std().values; z = doi / sd
    dp = (c - sh(c, w)) / a
    for zt in (1.5, 2.5):
        sig[f'YENİ{w}_{zt}'] = np.where((z >= zt) & (dp >= 1), 1, np.where((z >= zt) & (dp <= -1), -1, 0))        # yeni para yönünde
        sig[f'TASF{w}_{zt}'] = np.where((z <= -zt) & (dp <= -1), 1, np.where((z <= -zt) & (dp >= 1), -1, 0))      # tasfiye/sıkışma → ters
        sig[f'EMİL{w}_{zt}'] = np.where((z >= zt) & (np.abs(dp) < 0.5), 1, 0) * np.sign(np.nan_to_num(-(c - sh(c, w * 4))))  # OI şişiyor fiyat durmuş → önceki hareketin tersi
for w in (30, 60):
    hh = pd.Series(c).rolling(w).max().values; ll = pd.Series(c).rolling(w).min().values; doi = lo - sh(lo, w)
    sig[f'DIV{w}'] = np.where((c >= hh) & (doi < -0.005), -1, np.where((c <= ll) & (doi < -0.005), 1, 0))       # OI düşerken yeni uç → ters
rows = []
for k, v in sig.items():
    v = np.nan_to_num(v)
    for side in (1, -1):
        for lab, val in (('UP', 1), ('DN', -1), ('YAT', 0)):
            m = (v == side) & (R == val)
            if m[:mid].sum() < 150 or m[mid:].sum() < 150: continue
            row = dict(kural=k, yön='L' if side > 0 else 'S', rejim=lab, n=int(m.sum()))
            for h in (15, 60, 240):
                x = fw[h] * side; row[f'h{h}IS'] = np.nanmean(x[:mid][m[:mid]]); row[f'h{h}OOS'] = np.nanmean(x[mid:][m[mid:]])
            x = fw[60] * side; row['yıl+'] = sum(np.nanmean(x[(yr == y) & m]) > 0 for y in range(2022, 2027) if ((yr == y) & m).sum() > 50)
            rows.append(row)
D = pd.DataFrame(rows); D.to_csv('/home/claude/zecwork/micro1.csv', index=False)
print('toplam aday', len(D))
g = D[(D.h60IS >= 10) & (D.h60OOS > 0)].sort_values('h60IS', ascending=False)
print('IS h60 ≥10 ve OOS > 0 olanlar:', len(g), '/', (D.h60IS >= 10).sum(), 'IS seçili'); print(g.head(25).round(1).to_string(index=False))
print('\nIS-seçili ama OOS ≤ 0 (seçimin bedeli):'); print(D[(D.h60IS >= 10) & (D.h60OOS <= 0)].sort_values('h60IS', ascending=False).head(10).round(1).to_string(index=False))
