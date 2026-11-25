exec(open('/home/claude/zecwork/pfeval.py').read().split("rowsC, rowsP = [], []")[0])
ps_, ep_, xp_, xd_ = base_arrays(B2['TP'])
e, P, nt = ov3(o, h, l, c, ps_, ep_, xp_, xd_, R10 >= 94, R10 <= 6, R10 <= 70, R10 >= 30, 240, 0.03, 99.0, trSh, True, 0.02, TICK)   # V2.9
zs = pd.Series(e, ix_); z10 = zs.reindex(Pp.index, method='ffill')
Z9 = np.load('/home/claude/zecwork/base28.npz'); btc = pd.Series(Z9['btc'], ix_).ffill(); xmr = pd.Series(Z9['xmr'], ix_).ffill(); zc = pd.Series(c, ix_)
eth = pd.read_csv('/home/claude/peer/eth/ETHUSDT_1h.csv.gz'); eth = pd.Series(eth.close.values, pd.to_datetime(eth.time, unit='ms'))
pos_ = pd.Series(ps_, ix_)
rows = []
for a, b, d in legs:
    if a < W['son300']: continue
    g = lambda s: (s.asof(b) / s.asof(a) - 1) * 100
    sl = pos_[a:b]; rows.append(dict(baş=a.strftime('%m-%d %H:%M'), saat=round((b - a).total_seconds() / 3600, 1), portföy=-d, ZEC_pnl=z10.asof(b) - z10.asof(a), ZEC=g(zc), BTC=g(btc), ETH=g(eth), XMR=g(xmr),
                                  long_oran=(sl > 0).mean() * 100, short_oran=(sl < 0).mean() * 100))
D = pd.DataFrame(rows); pd.set_option('display.width', 220); print(D.round(1).to_string(index=False))
print('\nözet: ZEC kaybettiği çukurlar:', (D.ZEC_pnl < 0).sum(), '/', len(D), '· ZEC kayıp toplam', round(D.ZEC_pnl[D.ZEC_pnl < 0].sum()), '· kazanç', round(D.ZEC_pnl[D.ZEC_pnl > 0].sum()))
for k in ('BTC', 'ETH', 'XMR', 'ZEC'): print(k, 'çukurda ort. %', round(D[k].mean(), 2), '· ZEC_pnl ile kor.', round(D[k].corr(D.ZEC_pnl), 2))
print('ZEC kaybedilen çukurlarda short oranı ort.', round(D[D.ZEC_pnl < 0].short_oran.mean()), '· long', round(D[D.ZEC_pnl < 0].long_oran.mean()))
