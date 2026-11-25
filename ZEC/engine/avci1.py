# AVCI: 1 saatlik çapraz OI haritası (fiyat × OI birikimi, z) → flat iken harita yönünde yeniden giriş; V2.8 tabanı
exec(open('/home/claude/zecwork/trend2.py').read().split("eq, T = sim(X0); ad('V2.7 baz', eq, T); out[-1]")[0])
stt = state2(200, 55, 20, 0.15, 4); XB = [x.copy() for x in X0]; XB[1][1:5] &= ~(stt > 0)[None, :]; XB[3][1:5] &= ~(stt > 0)[None, :]
def flatp(T):
    held = (T[:, 1] - T[:, 0]).sum(); return 100 - held / (T[-1, 1] - T[0, 0]) * 100
def simA(AL_, AS_, th):
    z = np.zeros(n, np.bool_); rows = [tuple(XB[q][r] for q in range(4)) for r in range(0, 5)] + [(AL_, AS_, AL_, AS_), tuple(XB[q][5] for q in range(4))]
    X = [np.ascontiguousarray(np.vstack([r_[q] for r_ in rows])) for q in range(4)]
    return z03.simN11(o, h, l, c, a, *X, np.r_[1e6, TS6[1:5], 28.0, 8.0], np.r_[480, TH6[1:5], th, 480], spk, 60, 3, np.array([0, 0, 0, 0, 0, 0, 1], np.bool_), 0.10, False, F0, F0, np.r_[0.0, [0.10] * 4, 0.10, 0.10])
eq, T = sim(XB); ad('V2.8 baz', eq, T); out[-1].update(flat=flatp(T))
Op = Hh.oi_zec; Pp = Ch
BoiH = pd.Series(boi, idx).resample('1h').last() if 'boi' in dir() else None
X_ = pd.read_csv('/home/claude/peer/BTCUSDT_OI_5m.csv.gz', usecols=['timestamp_utc', 'open_interest_qty']); bOI = pd.Series(X_.open_interest_qty.values, pd.to_datetime(X_.timestamp_utc).dt.tz_localize(None)).resample('1h').last()
bPx = pd.Series(np.nan, Hh.index)
emaH = Pp.ewm(span=200 * 4, adjust=False).mean()
for k in (1, 4, 12):
    zP = Pp.pct_change(k) / Pp.pct_change(k).rolling(720, min_periods=240).std(); zO = Op.pct_change(k) / Op.pct_change(k).rolling(720, min_periods=240).std()
    zB = bOI.pct_change(k) / bOI.pct_change(k).rolling(720, min_periods=240).std(); zB = zB.reindex(zP.index)
    for thz in (0.5, 1.0):
        mp = pd.Series(np.where((zP >= thz) & (zO >= thz), 1, np.where((zP <= -thz) & (zO >= thz), -1, 0)), zP.index)
        mpB = pd.Series(np.where((mp != 0) & (zB >= 0), mp, 0), zP.index)                     # çapraz: BTC OI de artıyor
        for mn, mm in (('ZEC', mp), ('ZEC×BTC OI', mpB)):
            v = np.nan_to_num(map1m(mm.astype(float), 1)); up = np.nan_to_num(map1m((Pp > emaH).astype(float), 1)) > 0
            for fn, (al, as_) in (('filtresiz', (v > 0, v < 0)), ('4s trend yönü', ((v > 0) & up, (v < 0) & ~up))):
                for th in (60, 240):
                    eq, T = simA(al.astype(np.bool_), as_.astype(np.bool_), th); ad(f'AVCI {mn} {k}s z≥{thz} {fn} tut{th}', eq, T)
                    out[-1].update(flat=flatp(T), avciN=int((T[:, 7] == 5).sum()), avci=T[T[:, 7] == 5, 5].sum())
eq, T = simA((stt > 0).astype(np.bool_), np.zeros(n, np.bool_), 60); ad('AVCI = 4s ara trend long (oybirliği şartlı)', eq, T); out[-1].update(flat=flatp(T), avciN=int((T[:, 7] == 5).sum()), avci=T[T[:, 7] == 5, 5].sum())
R = pd.DataFrame(out); R.to_csv('/home/claude/pine_hello_world/ZEC/research/AVCI_OI_HARITA_1DK.csv', index=False)
cols = ['model', 'n', 'net', 'PF', 'MaxDD', 'DDp99', 'IS', 'OOS', 'TVpen', 'son300', 'R2_2023p', 'flat', 'avciN', 'avci', 'yıllar']; pd.set_option('display.width', 330)
print(R[[c_ for c_ in cols if c_ in R]].sort_values('net', ascending=False).head(14).round(1).to_string(index=False))
