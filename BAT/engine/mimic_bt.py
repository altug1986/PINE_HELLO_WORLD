import numpy as np, pandas as pd, v55
from short_top_lib import DIVh, zh, to5, q0, D, simR, run, evalp, m30, n, t
from met import SPLIT
import short_top_lib as S
X = pd.read_pickle('mimic_X.pkl'); P720 = pd.read_pickle('mimic_z720.pkl'); P24 = pd.read_pickle('mimic_z24.pkl')
sp = pd.Timestamp(SPLIT, unit='ms')
def h5(s):
    hk = ((s.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')).to_numpy(np.int64); j = np.searchsorted(hk, (t // 3600000) * 3600000) - 1
    return np.where(j >= 0, s.values[np.clip(j, 0, None)], np.nan)
def qgate(s, qhi):                       # eşik yalnız İLK dönemden: üst/alt yüzdelik
    a = s[s.index < sp].dropna(); hi, lo = a.quantile(qhi), a.quantile(1 - qhi); v = np.nan_to_num(h5(s), nan=(hi + lo) / 2); return v >= hi, v <= lo
def bt(GL, GS, lab):
    v55.GL = GL.astype(np.int64); v55.GS = GS.astype(np.int64); e = evalp(q0); m = m30(simR(D, run(q0)[0]))
    return dict(t=lab, net=e['net'], n=e['n'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], MaxDD30=m['MaxDD 30dk mum içi'], Calmar30=m['Calmar 30dk'])
Z = np.zeros(n, bool); rows = [bt(Z, Z, 'V5.7 (kapı yok)')]
a = np.nan_to_num(to5(zh(DIVh, 720))); b = np.nan_to_num(to5(zh(DIVh, 24))); rows.append(bt((a >= 1) | (b >= 1.5), (a <= -1) | (b <= -1.5), 'GERÇEK balina kapısı (V5.9)'))
for q in (0.80, 0.84, 0.90):
    L1, S1 = qgate(P720, q); rows.append(bt(L1, S1, f'taklit 30g (doğrusal) üst %{round((1-q)*100)}'))
    L2, S2 = qgate(P24, min(q + 0.09, 0.97)); rows.append(bt(L1 | L2, S1 | S2, f'taklit 30g + 24s üst %{round((1-q)*100)}'))
    L3, S3 = qgate(X['oi720'], q); rows.append(bt(L3, S3, f'sadece BAT OI 30 gün değişimi üst %{round((1-q)*100)}'))
d = pd.DataFrame(rows); pd.set_option('display.width', 220); print(d.to_string(index=False))
