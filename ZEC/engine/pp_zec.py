# ERR taşınabilirlik: PEPE V11.7 (ayarlar değişmeden) → ZEC 5m
exec(open('/home/claude/zecwork/pp_run.py').read())
M = pd.concat([pd.read_parquet(f, columns=['time', 'open', 'high', 'low', 'close', 'oi_zec']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))])
M.index = pd.to_datetime(M.time, unit='ms') if M.time.dtype.kind in 'iu' else pd.to_datetime(M.time); M = M[~M.index.duplicated()].sort_index()
Z = M[['open', 'high', 'low', 'close']].copy(); Z.columns = list('ohlc')
d5, A = pp.build(Z, M.oi_zec, btc, boi); np.save('/home/claude/zecwork/pp_zec_A.npy', A, allow_pickle=True); d5.to_pickle('/home/claude/zecwork/pp_zec_d5.pkl')
rows = []
eq, T = pp.run(A, tick=0.01); rows.append(metrics('ZEC 5m · PEPE V11.7 ayarları (değişmeden)', d5, eq, T, '2023-12-11')); np.save('/home/claude/zecwork/eq_PPZ.npy', eq); np.save('/home/claude/zecwork/T_PPZ.npy', T)
P = list(pp.PEPE_P); P[-1] = False
eq, T = pp.run(A, tuple(P), tick=0.01); rows.append(metrics('… BTC+ZEC ortak OI kuralları kapalı', d5, eq, T, '2023-12-11'))
pd.set_option('display.width', 300); print(pd.DataFrame(rows).round(2).to_string(index=False))
x = pd.Series(eq, d5.index); print('aylık pozitif %', round((x.resample('ME').last().diff() > 0).mean() * 100))
