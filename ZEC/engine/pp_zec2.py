# Oynaklık uyarlı taşıma: fiyat-% eşikleri ZEC/PEPE 5m oynaklık oranıyla ölçeklenir (arama yok; oran ×0.75/×1/×1.25 komşuluk)
exec(open('/home/claude/zecwork/pp_run.py').read())
A = np.load('/home/claude/zecwork/pp_zec_A.npy', allow_pickle=True).item(); d5 = pd.read_pickle('/home/claude/zecwork/pp_zec_d5.pkl')
pe = ld1m(sorted(glob.glob('/home/claude/peer/1000PEPEUSDT_1m_FULL_*.csv.gz'))).c.resample('5min').last().dropna()
zc = pd.Series(A['c'], d5.index); a0, a1 = '2023-06-01', '2026-11-15'
vp = pe[a0:a1].pct_change().abs().mean(); vz = zc[a0:a1].pct_change().abs().mean(); k = vz / vp
print('5 dk ort. |getiri| PEPE %', round(vp * 100, 3), '· ZEC %', round(vz * 100, 3), '· oran', round(k, 3))
yz = zc.pct_change().abs().groupby(zc.index.year).mean() * 100; print('ZEC yıllık 5dk |getiri| %:', yz.round(3).to_dict())
PX = [3, 6, 9, 11, 19, 21, 27, 29, 32, 35, 39]
rows = []
for m_ in (0.75, 1.0, 1.25):
    P = list(pp.PEPE_P)
    for j in PX: P[j] = P[j] * k * m_
    eq, T = pp.run(A, tuple(P), tick=0.01); rows.append(metrics(f'ZEC · PEPE V11.7 eşikler × {k * m_:.2f}', d5, eq, T, '2023-12-11'))
pd.set_option('display.width', 300); print(pd.DataFrame(rows).round(2).to_string(index=False))
