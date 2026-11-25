# Port doğrulama: PEPE 5m üzerinde V11.7 → TV export (pepe117.xlsx) ile kıyas
exec(open('/home/claude/zecwork/pp_run.py').read())
pe = ld1m(sorted(glob.glob('/home/claude/peer/1000PEPEUSDT_1m_FULL_*.csv.gz')) + sorted(glob.glob('/home/claude/peer/R_1000PEPEUSDT_*')))
poi = ldoi('/home/claude/peer/PEPE_OI_5m.csv.gz'); poi = poi[~poi.index.duplicated()]
d5, A = pp.build(pe, poi.reindex(pe.index, method='ffill'), btc, boi)
eq, T = pp.run(A, tick=0.0000001); print(pd.DataFrame([metrics('PEPE port V11.7', d5, eq, T)]).round(1).to_string(index=False))
x = pd.ExcelFile('/home/claude/pepeeng/pepe117.xlsx'); print(x.sheet_names)
d = x.parse('İşlemler'); d = d[d['Tarih ve saat'].astype(str) != 'Açık']; d['t'] = pd.to_datetime(d['Tarih ve saat']) - pd.Timedelta(hours=3)
en = d[d.Tip.str.contains('giriş', case=False)]; print('TV:', len(en), en.t.min(), '→', en.t.max(), 'net', round(en['Net K/Z USDT'].sum()))
t0 = en.t.min(); te = d5.index; et = te[T[:, 0].astype(int)]; m = (et >= t0) & (et <= en.t.max()); print('port aynı aralık:', m.sum(), 'net', round(T[m, 5].sum()))
ent = set(zip(et[m], T[m, 2].astype(int))); tv = set(zip(en.t, np.where(en.Tip.str.contains('Uzun'), 1, -1)))
print('giriş eşleşme % (zaman+yön):', round(len(ent & tv) / len(tv) * 100, 1))
