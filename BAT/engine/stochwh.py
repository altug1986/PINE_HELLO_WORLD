import numpy as np, pandas as pd, itertools, v55
from short_top_lib import d, DIVh, zh, to5, q0, D, simR, run, evalp, m30, n, t
from eng import rsi
x = d[(d.count_toptrader_long_short_ratio > 0) & (d.sum_toptrader_long_short_ratio > 0)]
ix = pd.to_datetime(x.ms, unit='ms')
SER = dict(POZ=pd.Series(x.sum_toptrader_long_short_ratio.values, index=ix), HES=pd.Series(x.count_toptrader_long_short_ratio.values, index=ix))
SER['FARK'] = np.log(SER['POZ']) - np.log(SER['HES'])
def stochrsi(s, lr=14, ls=14, sk=3, sd=3):
    r = pd.Series(rsi(s.values.astype(float), lr), index=s.index); lo = r.rolling(ls).min(); hi = r.rolling(ls).max()
    st = 100 * (r - lo) / (hi - lo); k = st.rolling(sk).mean(); return k, k.rolling(sd).mean()
def to5m(s, ms):                                   # önceki TAMAMLANMIŞ mumun değeri
    hk = ((s.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')).to_numpy(np.int64); j = np.searchsorted(hk, (t // ms) * ms) - 1
    return np.where(j >= 0, s.values[np.clip(j, 0, None)], np.nan)
a = np.nan_to_num(to5(zh(DIVh, 720))); b = np.nan_to_num(to5(zh(DIVh, 24))); WL = (a >= 1) | (b >= 1.5); WS = (a <= -1) | (b <= -1.5)
Z = np.zeros(n, bool)
def bt(GL, GS, lab):
    v55.GL = GL.astype(np.int64); v55.GS = GS.astype(np.int64); e = evalp(q0); m = m30(simR(D, run(q0)[0]))
    return dict(t=lab, net=e['net'], n=e['n'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L200=e['L200'], MaxDD30=m['MaxDD 30dk mum içi'], Calmar30=m['Calmar 30dk'])
if __name__ == "__main__":
    rows = [bt(Z, Z, 'V5.7 (kapı yok)'), bt(WL, WS, 'V6.1 (balina kapısı)')]
    for nm, tf in itertools.product(['POZ', 'HES', 'FARK'], ['30min']):
        s30 = SER[nm].resample(tf, label='left').last().dropna(); k, dd = stochrsi(s30); K = np.nan_to_num(to5m(k, 1800000), nan=50.0)
        for hi, md in itertools.product([80, 90], ['bizimle', 'karşı']):
            lo = 100 - hi
            GL, GS = (K >= hi, K <= lo) if md == 'bizimle' else (K <= lo, K >= hi)
            rows.append(bt(GL, GS, f'StochRSI30 {nm} {md} {hi}/{lo} (tek başına)'))
            rows.append(bt(WL | GL, WS | GS, f'V6.1 + StochRSI30 {nm} {md} {hi}/{lo}'))
    r = pd.DataFrame(rows); r.to_pickle('stochwh.pkl'); pd.set_option('display.width', 220)
    print(r.iloc[:2].to_string(index=False)); b0, b1 = r.iloc[0], r.iloc[1]
    print('tek başına V5.7\'den iyi (net+OOS+L200):', ((r.t.str.contains('tek')) & (r.net > b0.net) & (r.OOS > b0.OOS) & (r.L200 > b0.L200)).sum(), '| V6.1 üstüne V6.1\'den iyi:', ((r.t.str.startswith('V6.1 +')) & (r.net > b1.net) & (r.OOS > b1.OOS) & (r.L200 > b1.L200)).sum())
    print(r.iloc[2:].sort_values('Calmar30', ascending=False).head(8).to_string(index=False))
    