import numpy as np, pandas as pd, v55
from short_top_lib import to5, zh, DIVh, q0, D, simR, run, evalp, m30, n
h = DIVh.ffill()                                   # Pine: gaps_off → boş saatte son değer tekrar
st = h.first_valid_index(); hv = h[st:]
def zp(N):                                         # Pine dizisi: son N değer (geçerli başlangıçtan beri), en az N/2
    r = hv.rolling(N, min_periods=N // 2); z = ((hv - r.mean()) / r.std()).reindex(h.index).to_numpy(); return z
def go(z30, z24, lab):
    a = np.nan_to_num(to5(z30)); b = np.nan_to_num(to5(z24))
    v55.GL = ((a >= 1.0) | (b >= 1.5)).astype(np.int64); v55.GS = ((a <= -1.0) | (b <= -1.5)).astype(np.int64)
    e = evalp(q0); m = m30(simR(D, run(q0)[0]))
    print(f'{lab}: net {e["net"]} işlem {e["n"]} IS {e["IS"]} OOS {e["OOS"]} L12 {e["L12"]} L200 {e["L200"]} MaxDD30 {m["MaxDD 30dk mum içi"]} Calmar30 {m["Calmar 30dk"]} DD {m["DD P95/P98/P99 30dk"]}')
    return v55.GL.copy(), v55.GS.copy()
g1 = go(zh(DIVh, 720), zh(DIVh, 24), 'motor (takvim saati, boşluk hariç)')
g2 = go(zp(720), zp(24), 'Pine taklidi (boş saat = son değer)')
print('kapı durumu aynı olan bar %', round(((g1[0] == g2[0]) & (g1[1] == g2[1])).mean() * 100, 2))
# Pine algoritması 2: boş saatte TV son değeri tekrarlar → tekrar eden değer atlanır; pencere TAKVİM saatiyle (son N saat) kırpılır
hf = DIVh.ffill(); hd = hf.where(hf != hf.shift(1))
print('motor saat serisiyle aynı (NaN dahil) %', round(((hd == DIVh) | (hd.isna() & DIVh.isna())).mean() * 100, 3))
g3 = go(zh(hd, 720), zh(hd, 24), 'Pine taklidi 2 (tekrar atla + takvim penceresi)')
print('kapı durumu motorla aynı bar %', round(((g1[0] == g3[0]) & (g1[1] == g3[1])).mean() * 100, 3))
