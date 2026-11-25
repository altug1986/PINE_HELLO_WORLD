# YATAY MOD — büyük ölçek (15 dk / 1 saat sinyaller) + olay çalışması (sinyalden sonra fiyat baz yönde ne yapıyor)
src = open('/home/claude/zecwork/yatay.py').read(); exec(src.split("eq0 = np.empty(n)")[0])
eq0 = np.load('/home/claude/zecwork/eq_V29_yatay_baz.npy'); b6, bm = np.diff(eq0[B6]), np.diff(eq0[BM])
def rsiH(D): return P.rsi(D.c.values, 14)
def ev(v, D, td): return mapH(np.asarray(v, float), D, td) > 0.5
EX2 = []
for nm, D_, td in (('15dk', D15, pd.Timedelta('15min')), ('1s', D1, pd.Timedelta('1h'))):
    r = rsiH(D_); rp = np.r_[50.0, r[:-1]]
    for th in (25.0, 30.0): EX2.append((f'RSI14 {nm} {th:g} geri dönüş', ev((rp < th) & (r >= th), D_, td), ev((rp > 100 - th) & (r <= 100 - th), D_, td)))
H1, L1, tr1 = trng(D1); C1h = D1.c.values; A1h = rma(tr1, 14); l2 = S(L1).rolling(2).min().values; h2 = S(H1).rolling(2).max().values
for N in (24, 48):
    LLh, HHh = S(L1).rolling(N).min().values, S(H1).rolling(N).max().values; Wh = HHh - LLh
    for z in (0.1, 0.2):
        for m in (0.5, 1.0):
            EX2.append((f'kanal 1s×{N} dip/tepe %{z * 100:g} + {m:g}ATR dönüş', ev((l2 <= LLh + z * Wh) & (C1h >= l2 + m * A1h), D1, pd.Timedelta('1h')), ev((h2 >= HHh - z * Wh) & (C1h <= h2 - m * A1h), D1, pd.Timedelta('1h'))))
LLd, HHd = rmin(l, 1440), rmax(h, 1440); PB2 = np.where(HHd - LLd > 0, (c - LLd) / np.maximum(HHd - LLd, 1e-12), 0.5)
DET2 = [DET[k] for k in (3, 4, 1, 7, 8)]; YAT2 = np.stack([d[1] for d in DET2]); EXS2 = np.stack([e[1] for e in EX2]); EXL2 = np.stack([e[2] for e in EX2])
TBV2 = np.array([960, 2880], np.int64)
CB2 = np.array([(yi, ei) + t for yi in range(len(DET2)) for ei in range(len(EX2)) for t in itertools.product(range(2), range(3), range(2), range(2), range(3))], np.int64)
R2 = grid(CB2, PRE, CONT, TBV2, GV, o, h, l, c, posA, bXp, bXd, R10, YAT2, EXS2, EXL2, PB2, TICK, mid, t23, t300, B6, b6, BM, bm)
cols = ['net', 'IS', 'OOS', 'yakın', 'son300', 'MaxDD', 'mddIS', 'mddOOS', 'Ulcer', 'blok+', 'blok', 'ay+', 'ay-', 'çıkış']
d2 = pd.DataFrame(R2, columns=cols); d2['tespit'] = [DET2[k][0] for k in CB2[:, 0]]; d2['çıkış_tipi'] = [EX2[k][0] for k in CB2[:, 1]]
d2['geriPB'] = PRE[CB2[:, 2]]; d2['devam'] = CONT[CB2[:, 3]]; d2['süre'] = TBV2[CB2[:, 4]]; d2['ters'] = CB2[:, 5]; d2['kârŞartı'] = GV[CB2[:, 6]]
for k in ('net', 'IS', 'OOS', 'MaxDD', 'Ulcer'): d2['d' + k] = d2[k] - {'net': eq0[-1], 'IS': eq0[mid], 'OOS': eq0[-1] - eq0[mid], 'MaxDD': 391.3, 'Ulcer': 26.11}[k]
d2.to_pickle('/home/claude/zecwork/yatay2.pkl'); v = d2; v['ikiYarı'] = (v.dIS > 0) & (v.dOOS > 0)
print(f"BÜYÜK ÖLÇEK {len(v)} varyant | dIS>0 %{(v.dIS>0).mean()*100:.1f} | dOOS>0 %{(v.dOOS>0).mean()*100:.1f} | iki yarı %{v.ikiYarı.mean()*100:.1f} | medyan dNet {v.dnet.median():.0f} | max dNet {v.dnet.max():.0f} | kapı {((v.net>=9605.6)&(v.MaxDD<=391.3)&(v.Ulcer<=26.11)).sum()}")
print(v.groupby('çıkış_tipi').agg(medNet=('dnet', 'median'), maxNet=('dnet', 'max'), medIS=('dIS', 'median'), medOOS=('dOOS', 'median'), ikiYarı=('ikiYarı', 'mean')).round(2).to_string())
# OLAY ÇALIŞMASI: yatayda baz pozisyon varken "kaymak" sinyali → sonraki 1/4/12/24 saatte fiyat baz yönde % kaç gidiyor (pozitif = trend sürüyor, çıkmak kaybettirir)
yv = DET[7][1]; Dn = posA.astype(float); rows = []
def fwd(idx, k): j = np.minimum(idx + k, n - 1); return ((c[j] / c[idx] - 1) * Dn[idx] * 100)
base_idx = np.flatnonzero(yv & (posA != 0))[::30]
rows.append(dict(olay='yatayda rastgele an (baz yönü tutmak)', adet=len(base_idx), **{f'{k // 60}s': fwd(base_idx, k).mean() for k in (60, 240, 720, 1440)}))
for nm, eS, eL in [EX[1], EX[10], EX[14], EX2[3], EX2[6]]:
    sig = yv & (((posA < 0) & eS) | ((posA > 0) & eL)); st = np.flatnonzero(sig & ~np.r_[False, sig[:-1]]); keep = []; last = -10**9
    for s_ in st:
        if s_ - last >= 60: keep.append(s_); last = s_
    keep = np.array(keep); rows.append(dict(olay=nm, adet=len(keep), **{f'{k // 60}s': fwd(keep, k).mean() for k in (60, 240, 720, 1440)}))
print('\nOLAY: CHOP 1s>55 yatayında, sinyal sonrası fiyatın BAZ YÖNDE % hareketi (ortalama)'); print(pd.DataFrame(rows).round(3).to_string(index=False))
