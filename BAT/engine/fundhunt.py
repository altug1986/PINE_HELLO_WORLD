import numpy as np, pandas as pd
from p1m import build, run, SPL, raw
from btcx import ent, b24, b1
from edge2 import sig
from rev25 import feats, apply25
D = build(5); c = D['c']; t = D['t']; n = len(c)
fe = pd.read_csv('fund_events.csv'); fe = fe[fe.rate_type == 'Regular'].sort_values('funding_time_ms'); ft = fe.funding_time_ms.to_numpy(); fr = fe.funding_rate_decimal.to_numpy() * 100   # %
print('funding olayı', len(fe), '| dağılım % (5/25/50/75/95):', np.round(np.percentile(fr, [5, 25, 50, 75, 95]), 4), '| = %0.01 oranı', round((np.abs(fr - 0.01) < 1e-9).mean() * 100), '%')
j = np.searchsorted(ft, t, side='right') - 1                                   # bar açılışında bilinen son kesim
F = np.where(j >= 0, fr[np.clip(j, 0, None)], np.nan); F3 = np.where(j >= 2, pd.Series(fr).rolling(3).mean().to_numpy()[np.clip(j, 0, None)], np.nan)
zf = (pd.Series(fr) - pd.Series(fr).rolling(90).mean()) / pd.Series(fr).rolling(90).std(); Fz = np.where(j >= 0, zf.to_numpy()[np.clip(j, 0, None)], np.nan)
prevneg = np.where(j >= 3, pd.Series((fr > 0).astype(float)).rolling(3).sum().shift(1).to_numpy()[np.clip(j, 0, None)], np.nan)
R_ = raw(); oT, oV = R_['oT'], R_['oV']; f5 = t // 300000 * 300000; jo = np.searchsorted(oT, f5 - 300000, side='right') - 1; oi = np.where(jo >= 0, oV[np.clip(jo, 0, None)], np.nan); oi[f5 - 300000 < oT[0]] = np.nan
L = lambda a, k: np.r_[np.full(k, np.nan), a[:-k]]; ch = lambda a, k: (a / L(a, k) - 1) * 100
p24 = ch(c, 288); o24 = ch(oi, 288)
# V7.1 pozisyonu (motor, balina tam)
Fr = feats(D, False); deep = Fr['deep']; cs = np.where(deep == -1, 0, sig('Kapitülasyon', (5, 4))).astype(np.int64); bs = np.where(deep == -1, 0, ((b24 <= -10) & (b1 >= 0))).astype(np.int64)
cap = ent(cs, c, 144, 5.0); side = np.where(cap != 0, cap, ent(bs, c, 288, 3.0)); pos = apply25(D, run(D, ovr=side), Fr)
E = {'F ≤ −%0.03 (short ödüyor)': F <= -0.03, 'F ≤ −%0.05': F <= -0.05, 'F ≤ −%0.10': F <= -0.10, 'F ≥ +%0.03 (long ödüyor)': F >= 0.03, 'F ≥ +%0.05': F >= 0.05, 'F ≥ +%0.10': F >= 0.10,
     '3 kesim ort. ≤ −%0.02': F3 <= -0.02, '3 kesim ort. ≥ +%0.03': F3 >= 0.03, 'F z30g ≤ −2': Fz <= -2, 'F z30g ≥ +2': Fz >= 2, 'İlk negatife dönüş (önceki 3 pozitif)': (F < 0) & (prevneg == 3),
     'Negatif F + 24s düşüş ≥ %5': (F < 0) & (p24 <= -5), 'Pozitif F≥%0.03 + 24s yükseliş ≥ %5': (F >= 0.03) & (p24 >= 5), 'Negatif F + OI 24s ≥ +%5 (short birikimi)': (F < 0) & (o24 >= 5),
     'F ≥ %0.03 + OI 24s ≥ +%5 (long birikimi)': (F >= 0.03) & (o24 >= 5), 'Negatif F + 24s yükseliş ≥ %3 (sıkışma başladı)': (F < 0) & (p24 >= 3)}
H = {'1s': 12, '4s': 48, '12s': 144, '24s': 288}; fw = {h: (L(c[::-1], k)[::-1] / c - 1) * 100 for h, k in H.items()}; valid = ~np.isnan(fw['24s']) & ~np.isnan(F); rng = np.random.default_rng(3); rows = []
pool = np.flatnonzero(valid)
for lab, cond in E.items():
    cond = np.nan_to_num(cond).astype(bool) & valid; ev = []; last = -10**9
    for i in np.flatnonzero(cond & ~np.r_[False, cond[:-1]]):
        if i - last >= 288: ev.append(i); last = i
    ev = np.array(ev, int); o = dict(olay=lab, n=len(ev))
    if len(ev) < 25: rows.append(o); continue
    for h in H:
        f = fw[h][ev]; mu = f.mean(); null = np.array([fw[h][rng.choice(pool, len(ev))].mean() for _ in range(2000)]); o[f'{h} %'] = round(mu, 2); o[f'{h} p'] = round((np.sum(np.abs(null - null.mean()) >= abs(mu - null.mean())) + 1) / 2001, 3)
    f = fw['24s'][ev]; isv = t[ev] < SPL; o['24s ilk/son'] = f'{f[isv].mean():+.2f} / {f[~isv].mean():+.2f}' if (~isv).sum() else f'{f[isv].mean():+.2f} / —'
    o['V7.1 aynı yön %'] = round((pos[ev] == np.sign(f.mean())).mean() * 100); rows.append(o)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 30); print(pd.DataFrame(rows).to_string(index=False))
