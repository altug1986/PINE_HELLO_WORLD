# D: beta-düzeltilmiş ayrışma (ZEC vs BTC/ETH/XMR, saatlik kapanmış bar) → devam mı ortalamaya dönüş mü (IS/OOS + monkey)
import numpy as np, pandas as pd, glob
z = pd.concat([pd.read_parquet(f, columns=['time', 'close']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))])
z.index = pd.to_datetime(z.time, unit='ms') if z.time.dtype.kind in 'iu' else pd.to_datetime(z.time); Z = z.close.resample('1h').last()
def peer(pat):
    fs = sorted(glob.glob(f'/home/claude/peer/{pat}')); d = pd.concat([pd.read_csv(f) for f in fs]); t = d.columns[0]
    d.index = pd.to_datetime(d[t].astype('int64'), unit='ms') if d[t].dtype.kind in 'iuf' else pd.to_datetime(d[t]); d = d[~d.index.duplicated()].sort_index(); return d['close'].resample('1h').last()
BTC = peer('BTCUSDT_PERP_1m_*'); XMR = peer('XMRUSDT_1m_FULL*'); e = pd.read_csv('/home/claude/peer/eth/ETHUSDT_1h.csv.gz'); ETH = pd.Series(e.close.values, pd.to_datetime(e.time, unit='ms'))
D = pd.concat({'Z': Z, 'BTC': BTC, 'ETH': ETH, 'XMR': XMR}, axis=1).dropna(); r = np.log(D).diff()
mid = D.index[len(D) // 2]; print('saat:', len(D), D.index[0], '→', D.index[-1], '· orta', mid)
rows = []; rng = np.random.default_rng(5); cost = 8.5
for mk in ('BTC', 'ETH', 'XMR', 'piyasa'):
    m = r[['BTC', 'ETH', 'XMR']].mean(1) if mk == 'piyasa' else r[mk]
    beta = r.Z.rolling(720, min_periods=240).cov(m) / m.rolling(720, min_periods=240).var(); res = r.Z - beta.shift(1) * m
    for L in (4, 24):
        s = res.rolling(L).sum(); zs = (s - s.rolling(720, min_periods=240).mean()) / s.rolling(720, min_periods=240).std()
        for H in (4, 24):
            fw = (np.log(D.Z).shift(-H) - np.log(D.Z)) * 1e4     # kapanıştan sonraki H saat (sinyal kapanmış barda)
            for th in (2.0, 2.5):
                ev = zs.abs() >= th; ev = ev & ~ev.shift(1, fill_value=False)
                sg = np.sign(zs[ev]); f = fw[ev]; ok = f.notna(); sg, f = sg[ok], f[ok]
                dev = f * sg - cost; rev = -f * sg - cost; i_ = f.index < mid
                mkr = np.array([(fw.dropna().sample(len(f), random_state=int(k)).values * rng.choice([-1, 1], len(f))).mean() for k in range(200)])
                best = 'devam' if dev.mean() > rev.mean() else 'dönüş'; x = dev if best == 'devam' else rev
                rows.append(dict(piyasa=mk, pencere=f'{L}s', ufuk=f'{H}s', z=th, n=len(f), hipotez=best, ortBps=x.mean(), IS=x[i_].mean(), OOS=x[~i_].mean(), IS_n=i_.sum(),
                                 monkeyP=(mkr - cost >= x.mean()).mean(), net500=(x * 0.05).sum()))
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/BETA_AYRISMA_1S.csv', index=False); pd.set_option('display.width', 220)
print(R.round(2).to_string(index=False)); g = R[(R.IS > 0) & (R.OOS > 0) & (R.monkeyP < 0.05)]; print('\ngeçen:', len(g), '/', len(R))
