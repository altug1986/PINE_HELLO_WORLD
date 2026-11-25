# V2.8 baz: bar bazında pozisyon + equity + ham diziler → npz (hızlı tarama için)
exec(open('/home/claude/zecwork/trend2.py').read().split("eq, T = sim(X0); ad('V2.7 baz', eq, T); out[-1]")[0])
stt = state2(200, 55, 20, 0.15, 4); XB = [x.copy() for x in X0]; XB[1][1:5] &= ~(stt > 0)[None, :]; XB[3][1:5] &= ~(stt > 0)[None, :]
eq, T = sim(XB); ad('V2.8 baz', eq, T)
posA = np.zeros(n, np.int8)
for r in T: posA[int(r[0]):int(r[1])] = int(r[2])        # giriş barı kapanışından çıkış barı kapanışına kadar
import glob
MM = pd.concat([pd.read_parquet(f, columns=['time', 'volume', 'basis_bps', 'funding_pct', 'oi_zec', 'btc_close', 'xmr_close']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))])
MM.index = pd.to_datetime(MM.time, unit='ms') if MM.time.dtype.kind in 'iu' else pd.to_datetime(MM.time); MM = MM[~MM.index.duplicated()].reindex(idx)
f5 = idx.floor('5min') - pd.Timedelta(minutes=5); oiL = MM.oi_zec.reindex(f5).values          # kapanmış önceki 5 dk OI
tms = np.asarray((idx - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'), np.int64)
np.savez_compressed('/home/claude/zecwork/base28.npz', o=o, h=h, l=l, c=c, a=a, eq=eq, T=T, posA=posA, tms=tms, mid=mid, vol=MM.volume.values, basis=MM.basis_bps.values,
                    fund=MM.funding_pct.values, oi=oiL, btc=MM.btc_close.values, xmr=MM.xmr_close.values, spk=spk, stt=stt, pmp=pmp)
print('kaydedildi', n, 'bar · baz pozisyonda %', round((posA != 0).mean() * 100, 1), '· NaN btc %', round(np.isnan(MM.btc_close.values).mean() * 100, 2), '· oi %', round(np.isnan(oiL).mean() * 100, 1))
