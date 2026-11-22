import numpy as np, pandas as pd
from fundhunt import D, c, t, run, pos, side, Fr, apply25, L, ch, oi, fw, H, SPL
from edge2 import bc
x = pd.read_csv('btcoi5m.csv.gz', usecols=['source_create_time_utc', 'open_interest_qty']); xt = pd.to_datetime(x.source_create_time_utc).to_numpy().astype('datetime64[ms]').astype(np.int64)
xv = x.open_interest_qty.to_numpy(float); m = xv > 0; xt, xv = xt[m], xv[m]; o = np.argsort(xt); xt, xv = xt[o], xv[o]
f5 = t // 300000 * 300000; jx = np.searchsorted(xt, f5 - 300000, side='right') - 1; boi = np.where(jx >= 0, xv[np.clip(jx, 0, None)], np.nan); boi[f5 - 300000 < xt[0]] = np.nan
bo6, bo48, bo288 = ch(boi, 6), ch(boi, 48), ch(boi, 288); bp6, bp48, bp288 = ch(bc, 6), ch(bc, 48), ch(bc, 288); ao48, ap48 = ch(oi, 48), ch(c, 48)
E = {'BTC kapitülasyon (BTC OI 4s ≤ −3, BTC 4s ≤ −3)': (bo48 <= -3) & (bp48 <= -3), 'BTC kapitülasyon sert (−5 / −5)': (bo48 <= -5) & (bp48 <= -5),
     'BTC likidasyon 30 dk (OI ≤ −1.5, fiyat ≤ −1.5)': (bo6 <= -1.5) & (bp6 <= -1.5), 'BTC short sıkışması (OI 4s ≤ −3, fiyat 4s ≥ +3)': (bo48 <= -3) & (bp48 >= 3),
     'BTC OI boşalması 24s ≤ −10': bo288 <= -10, 'BTC kaldıraç birikimi + yükseliş (OI 24s ≥ 8, fiyat ≥ 4)': (bo288 >= 8) & (bp288 >= 4),
     'BTC kaldıraç birikimi + düşüş (OI 24s ≥ 8, fiyat ≤ −4)': (bo288 >= 8) & (bp288 <= -4),
     'BTC kapitülasyon + BAT düşüşü (BAT 4s ≤ −4)': (bo48 <= -3) & (bp48 <= -3) & (ap48 <= -4), 'BAT kapitülasyonu (karşılaştırma)': (ao48 <= -5) & (ap48 <= -4)}
valid = ~np.isnan(fw['24s']); rng = np.random.default_rng(5); rows = []; pool = np.flatnonzero(valid & ~np.isnan(boi))
for lab, cond in E.items():
    cond = np.nan_to_num(cond).astype(bool) & valid; ev = []; last = -10**9
    for i in np.flatnonzero(cond & ~np.r_[False, cond[:-1]]):
        if i - last >= 288: ev.append(i); last = i
    ev = np.array(ev, int); o = dict(olay=lab, n=len(ev))
    if len(ev) < 25: rows.append(o); continue
    for h in H:
        f = fw[h][ev]; mu = f.mean(); null = np.array([fw[h][rng.choice(pool, len(ev))].mean() for _ in range(2000)]); o[f'{h} %'] = round(mu, 2); o[f'{h} p'] = round((np.sum(np.abs(null - null.mean()) >= abs(mu - null.mean())) + 1) / 2001, 3)
    f = fw['24s'][ev]; isv = t[ev] < SPL; o['24s ilk/son'] = f'{f[isv].mean():+.2f} / {f[~isv].mean():+.2f}'; o['V7.1 aynı yön %'] = round((pos[ev] == np.sign(f.mean())).mean() * 100); rows.append(o)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 30); print(pd.DataFrame(rows).to_string(index=False))
