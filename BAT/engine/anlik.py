import numpy as np, pandas as pd, pickle
from p1m import build, run, trades
from eng import rma, trr
D = build(5); c = D['c']; o_ = None; h = D['h']; l = D['l']; n = len(c)
tr = trr(h, l, c, True); atr = rma(tr, 14); r = np.r_[0, c[1:] / c[:-1] - 1] * 100              # 5 dk mum getirisi
R, _ = trades(D, run(D)); eb, xb, d, q = R['eb'], R['xb'], R['dir'], R['q']
print(f"V6.6: {len(eb)} işlem, net {R['pnl'].sum():.0f}$")
for B in (2.0, 3.0, 4.0):
    big = np.abs(r) >= B
    ge = d * (c[eb] - c[eb - 1]) * q; gx = d * (c[xb - 1] - c[xb]) * q          # bir mum önce girseydik / çıksaydık farkı
    me = big[eb]; mx = big[xb]; withE = me & (np.sign(r[eb]) == d); agE = me & (np.sign(r[eb]) == -d); agX = mx & (np.sign(r[xb]) == -d); wX = mx & (np.sign(r[xb]) == d)
    print(f'\n|mum| ≥ %{B:g}: tüm mumların %{big.mean()*100:.2f}\'i')
    print(f'  GİRİŞ büyük mumda {me.sum()} | mum pozisyon yönüne (geç kalınmış) {withE.sum()}: bir önce girmek {ge[withE].sum():+.0f}$ | mum ters yöne (dip/tepe avı) {agE.sum()}: bir önce girmek {ge[agE].sum():+.0f}$')
    print(f'  ÇIKIŞ büyük mumda {mx.sum()} | mum pozisyona ters (şelale) {agX.sum()}: bir önce çıkmak {gx[agX].sum():+.0f}$ | mum lehte {wX.sum()}: bir önce çıkmak {gx[wX].sum():+.0f}$')
# anlık ATR büyük mumu önceden haber veriyor mu?  önceki mumun TR / ATR(14)
x = np.r_[np.nan, tr[:-1] / np.r_[np.nan, atr[:-2]]] if False else np.r_[np.nan, (tr / np.r_[np.nan, atr[:-1]])[:-1]]
m1 = pickle.load(open('D1.pkl', 'rb')); t1 = m1['t']; c1 = m1['c']; t = D['t']
r1 = np.abs(np.r_[0, c1[1:] / c1[:-1] - 1]) * 100; a1 = pd.Series(r1).rolling(60).mean().to_numpy()        # 1 dk: son 1 dk hareketi / son 60 dk ortalaması
j = np.searchsorted(t1, t + 300000) - 1; j2 = np.searchsorted(t1, t) - 1                                 # 5 dk mumun son 1 dk'sı; bir önceki mumun son 1 dk'sı
y1 = np.r_[np.nan, (r1[j] / a1[j])[:-1]]
print('\nsonraki 5 dk mumun |getiri| ≥ %3 olma olasılığı:')
base = (np.abs(r) >= 3).mean(); print(f'  koşulsuz %{base*100:.2f}')
for k in (2, 3, 5):
    m = np.nan_to_num(x) >= k; print(f'  önceki 5 dk mum TR ≥ {k}×ATR14: %{(np.abs(r[m]) >= 3).mean()*100:.2f} (mumların %{m.mean()*100:.1f}\'i)', end=' |')
    m = np.nan_to_num(y1) >= k * 2; print(f' önceki mumun son 1 dk hareketi ≥ {k*2}× 60 dk ort.: %{(np.abs(r[m]) >= 3).mean()*100:.2f} (%{m.mean()*100:.1f})')
    big = np.abs(r) >= 3; s = np.sign(r); ps = np.r_[0, s[:-1]]; mm = (np.nan_to_num(x) >= k) & big; print(f'     → büyük mum önceki mumla aynı yönde: %{(s[mm] == ps[mm]).mean()*100:.0f}')
