"""CANLI SÜRÜŞ KALİTESİ (kapanıştan bağımsız, yön odaklı).
Her olayda girişten sonra bar içi yol (2 dk high/low):
  İLK DOKUNUŞ k: önce lehe k·ATR mi, aleyhe k·ATR mi? (aynı barda ikisi → aleyhe say, muhafazakâr) → P(lehe)
  DOĞRU TARAF: sonraki 60 barın kaçında kapanış girişin lehe tarafında (%)
  SÜRÜŞ: 60 bar boyunca ortalama MTM (bps, maliyetsiz) ve en kötü bar içi ters gidiş (MAE)."""
import numpy as np
from numba import njit


@njit(cache=True)
def path(h, l, c, atr, idx, side, ks, W):
    m = len(idx); nk = len(ks); P = np.full((m, nk), np.nan); right = np.full(m, np.nan); avg = np.full(m, np.nan); mae = np.full(m, np.nan)
    n = len(c)
    for e in range(m):
        i = idx[e]; ep = c[i]; a = atr[i]
        if i + W >= n or np.isnan(a):
            continue
        rs = 0.0; s = 0.0; worst = 0.0
        for j in range(i + 1, i + W + 1):
            r = (c[j] - ep) / ep * side; s += r
            if r > 0: rs += 1
            adv = (ep - l[j]) / ep if side == 1 else (h[j] - ep) / ep
            worst = max(worst, adv)
        right[e] = rs / W * 100; avg[e] = s / W * 1e4; mae[e] = worst * 100
        for q in range(nk):
            k = ks[q]; up = ep + side * k * a; dn = ep - side * k * a; res = np.nan
            for j in range(i + 1, min(n, i + 721)):
                fav = (h[j] >= up) if side == 1 else (l[j] <= up)
                bad = (l[j] <= dn) if side == 1 else (h[j] >= dn)
                if bad:
                    res = 0.0; break
                if fav:
                    res = 1.0; break
            P[e, q] = res
    return P, right, avg, mae
