import numpy as np, pandas as pd
from p1m import build, run, SPL, raw
D = build(5); c = D['c']; t = D['t']; n = len(c); pos = run(D)
R_ = raw(); oT, oV = R_['oT'], R_['oV']; f5 = t // 300000 * 300000
j = np.searchsorted(oT, f5 - 300000, side='right') - 1; oi = np.where(j >= 0, oV[np.clip(j, 0, None)], np.nan); oi[f5 - 300000 < oT[0]] = np.nan
L = lambda a, k: np.r_[np.full(k, np.nan), a[:-k]]
ch = lambda a, k: (a / L(a, k) - 1) * 100
z = lambda x, N: ((pd.Series(x) - pd.Series(x).rolling(N, min_periods=N // 2).mean()) / pd.Series(x).rolling(N, min_periods=N // 2).std()).to_numpy()
p1, p4, p24 = ch(c, 12), ch(c, 48), ch(c, 288); o1, o4, o24 = ch(oi, 12), ch(oi, 48), ch(oi, 288); oiz = z(np.log(oi), 8640)
m = pd.read_pickle('top5.pkl'); jm = np.searchsorted(m.ms.to_numpy(), f5 - 300000, side='right') - 1; mv = lambda col: np.where(jm >= 0, m[col].where(m[col] > 0).to_numpy()[np.clip(jm, 0, None)], np.nan)
tp = np.log(mv('sum_toptrader_long_short_ratio')); gl = np.log(mv('count_long_short_ratio')); tk = np.log(mv('sum_taker_long_short_vol_ratio'))
k = pd.read_csv('BATUSDT_1DK_OHLCV_UTC.csv', usecols=['open_time_ms', 'volume', 'number_of_trades']); a = k.groupby(k.open_time_ms // 300000)[['volume', 'number_of_trades']].sum().reindex(t // 300000).to_numpy()
v1 = pd.Series(a[:, 0]).rolling(12).sum().to_numpy(); vr = v1 / (pd.Series(a[:, 0]).rolling(288).sum().to_numpy() / 24); nr = pd.Series(a[:, 1]).rolling(12).sum().to_numpy() / (pd.Series(a[:, 1]).rolling(288).sum().to_numpy() / 24)
btc = np.asarray(D.get('btc', None)) if 'btc' in D else None
x = R_['x']; kx = (x[:, 0] // 300000).astype(np.int64); bc = pd.Series(x[:, 4], index=kx).groupby(level=0).last().reindex(t // 300000).ffill().to_numpy(); b24 = ch(bc, 288)
hi24 = c >= pd.Series(c).rolling(288).max().to_numpy(); lo24 = c <= pd.Series(c).rolling(288).min().to_numpy()
E = {'OI birikimi, fiyat yatay (OI24≥+10, |F24|<2)': (o24 >= 10) & (np.abs(p24) < 2), 'OI birikimi + yükseliş (OI24≥6, F24≥4) [8 nolu]': (o24 >= 6) & (p24 >= 4),
     'OI birikimi + düşüş (OI24≥6, F24≤−4) [8 nolu]': (o24 >= 6) & (p24 <= -4), 'OI boşalması (OI24≤−10)': o24 <= -10, 'Kapitülasyon (OI4≤−5, F4≤−4)': (o4 <= -5) & (p4 <= -4),
     'Short sıkışması (OI4≤−5, F4≥+4)': (o4 <= -5) & (p4 >= 4), 'Hızlı kaldıraç (OI1≥+4)': o1 >= 4, 'OI aşırı yüksek (z30≥2.5)': oiz >= 2.5, 'OI aşırı düşük (z30≤−2.5)': oiz <= -2.5,
     '24s tepe + OI düşüyor (short kapanış rallisi)': hi24 & (o24 <= -3), '24s dip + OI düşüyor (long tasfiyesi)': lo24 & (o24 <= -3), '24s dip + OI artıyor (dipte short yığılması)': lo24 & (o24 >= 5),
     '24s tepe + OI artıyor (tepede long yığılması)': hi24 & (o24 >= 5), 'Top trader poz. 24s z≥2': z(tp - L(tp, 288), 8640) >= 2, 'Top trader poz. 24s z≤−2': z(tp - L(tp, 288), 8640) <= -2,
     'Kalabalık long (genel L/S z30≥2)': z(gl, 8640) >= 2, 'Kalabalık short (genel L/S z30≤−2)': z(gl, 8640) <= -2, 'Taker alış baskısı, fiyat yatay (z≥2.5, |F1|<1)': (z(tk, 288) >= 2.5) & (np.abs(p1) < 1),
     'Taker satış baskısı, fiyat yatay (z≤−2.5, |F1|<1)': (z(tk, 288) <= -2.5) & (np.abs(p1) < 1), 'Hacim patlaması, dar fiyat (H1≥4×, |F1|<1.5)': (vr >= 4) & (np.abs(p1) < 1.5),
     'İşlem adedi patlaması, dar fiyat': (nr >= 4) & (np.abs(p1) < 1.5), 'BTC çöküşü (B24≤−8)': b24 <= -8, 'BTC sıçraması (B24≥+8)': b24 >= 8}
H = {'1s': 12, '4s': 48, '12s': 144, '24s': 288}; fw = {h: (L(c[::-1], k)[::-1] / c - 1) * 100 for h, k in H.items()}       # ileri getiri (long)
valid = ~np.isnan(fw['24s']); rng = np.random.default_rng(1); rows = []
for lab, cond in E.items():
    cond = np.nan_to_num(cond).astype(bool) & valid; ev = []; last = -10**9
    for i in np.flatnonzero(cond & ~np.r_[False, cond[:-1]]):
        if i - last >= 288: ev.append(i); last = i                                     # bağımsız olaylar: 24 s ara
    ev = np.array(ev, int)
    if len(ev) < 30: rows.append(dict(olay=lab, n=len(ev))); continue
    o = dict(olay=lab, n=len(ev))
    pool = np.flatnonzero(valid & ~np.isnan(oi)) if 'OI' in lab else np.flatnonzero(valid)
    for h in H:
        f = fw[h][ev]; mu = f.mean(); null = np.array([fw[h][rng.choice(pool, len(ev))].mean() for _ in range(2000)]); p = (np.sum(np.abs(null - null.mean()) >= abs(mu - null.mean())) + 1) / 2001
        o[f'{h} ort %'] = round(mu, 2); o[f'{h} p'] = round(p, 3)
    f = fw['24s'][ev]; isv = t[ev] < SPL; o['24s ilk/son dönem %'] = f'{f[isv].mean():+.2f} / {f[~isv].mean():+.2f}'; o['tutarlı'] = bool(np.sign(f[isv].mean()) == np.sign(f[~isv].mean()) and isv.sum() >= 10 and (~isv).sum() >= 10)
    o['V6.6 aynı yönde %'] = round((pos[ev] == np.sign(f.mean())).mean() * 100); rows.append(o)
r = pd.DataFrame(rows); r.to_pickle('edgehunt.pkl'); pd.set_option('display.width', 300); pd.set_option('display.max_columns', 30)
print(r.to_string(index=False))
