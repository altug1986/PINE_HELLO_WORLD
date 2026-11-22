import pickle, numpy as np, pandas as pd
from p1m import build, run, report
D5 = build(5); D1 = pickle.load(open('D1.pkl', 'rb')); t1 = D1['t']
j = np.searchsorted(D5['t'], t1 // 300000 * 300000 - 300000); ok = (j < len(D5['t'])) & (D5['t'][np.clip(j, 0, len(D5['t']) - 1)] == t1 // 300000 * 300000 - 300000); j = np.clip(j, 0, len(D5['t']) - 1)
def m(k):
    a = D5[k][j]; return np.where(ok, a, np.nan) if a.dtype.kind == 'f' else np.where(ok, a, 0).astype(a.dtype)
G = {'z-skoru (z48, z96)': ['z48', 'z96'], 'RSI (14, 7)': ['r14', 'r7'], '24 s getiri sinyalleri (d288, rr, OI 24 s, hacim)': ['d288', 'rr', 'dO288', 'vr', 'nrv'],
     'likidasyon (30 dk fiyat/OI)': ['dPx6', 'dOi6', 'okO'], 'BTC lider (15 dk)': ['br', 'ar']}
rows = [report(D1, run(D1), '1 dk saf')]
for lab, ks in G.items(): rows.append(report(D1, run(D1, X={k: m(k) for k in ks}), f'5 dk kaynak: {lab}'))
cr = np.where(ok, D5['c'][j], np.nan); cr = pd.Series(cr).ffill().bfill().to_numpy(); rows.append(report(D1, run(D1, X={'crev': cr}), '5 dk kaynak: %3/%8 dönüş (zigzag)'))
allX = {k: m(k) for ks in G.values() for k in ks}; allX['crev'] = cr; rows.append(report(D1, run(D1, X=allX), '5 dk kaynak: HEPSİ (1 dk sadece icra/stop)'))
rows.append(report(D5, run(D5), '5 dk V6.6 (referans)'))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
