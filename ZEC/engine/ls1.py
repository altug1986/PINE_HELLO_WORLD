# OI oturuşu + long/short bloklarıyla rejim (IS seçim), V2.7 tabanı (4 motor + POMPA + %10 tavan); portföy akış çukurları
exec(open('/home/claude/zecwork/flat1.py').read().split("eq, T = z03.simN3(")[0])
import importlib, glob; importlib.reload(z03); pd.set_option('display.width', 340)
M = pd.concat([pd.read_parquet(f, columns=['time', 'oi_zec', 'tt_pos_ls', 'glb_ls']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))])
M.index = pd.to_datetime(M.time, unit='ms') if M.time.dtype.kind in 'iu' else pd.to_datetime(M.time); M = M.drop(columns='time')
Hh = M.resample('1h').last(); Ch = pd.Series(c, idx).resample('1h').last()
def to1m(s):   # saatlik değer, saat kapanınca (+1 s) kullanılabilir
    s = s.shift(1); j = np.searchsorted(s.index.values, idx.values, side='right') - 1; v = s.values[np.maximum(j, 0)]; v[j < 0] = np.nan; return v
bitn = lambda x: (np.nan_to_num(x, nan=1.0) >= 0).astype(np.int64)
oi24 = Hh.oi_zec.pct_change(24); px24 = Ch.pct_change(24)
FEAT = {'OIbirikim': to1m(oi24 - oi24.rolling(720, min_periods=240).median()),                      # OI 24 s değişimi 30 g medyanının üstü
        'OIfiyatUyum': to1m(np.sign(oi24) * np.sign(px24)),                                            # fiyat ve OI aynı yön (pozisyon trendi besliyor)
        'LSglb': to1m(Hh.glb_ls - Hh.glb_ls.rolling(720, min_periods=240).median()),                  # kalabalık long tarafında mı
        'LStop': to1m(Hh.tt_pos_ls - Hh.tt_pos_ls.rolling(720, min_periods=240).median())}
cut = np.datetime64('2026-04-01'); TVmask = idx.values < cut
R0i = np.asarray(R0, np.int64)
def selF(R, thr, hz, od):
    nc = int(R.max()) + 1; L = np.zeros(n, np.float32); S = np.zeros(n, np.float32); f = F[hz]; Rt = R[:mid]; ft = f[:mid]; fv = ~np.isnan(ft)
    for k in od:
        vt = V[k][:mid]; va = V[k]
        for side in (1, -1):
            m = vt == side; ca = np.bincount(Rt[m], minlength=nc); mv = m & fv
            sm = np.bincount(Rt[mv], weights=ft[mv], minlength=nc); cv = np.bincount(Rt[mv], minlength=nc)
            ok = (ca >= 400) & (cv > 0) & (sm / np.maximum(cv, 1) * side >= thr); (L if side > 0 else S)[:] += (va == side) & ok[R]
    return L, S
CF = [(PR + OI + ['X17_basis'], 25, 480, 2, 2, 1, 0, 10, 0.4, 4.0, 2), (PR + OI, 10, 120, 2, 1, 0, 0, 10, 0.4, 99.0, 0),
      (PR + OI + ['X17_basis'], 25, 120, 2, 1, 0, 0, 10, 0.0, 99.0, 0), (PR + OI + EXT, 25, 240, 2, 2, 0, 0, 10, 0.4, 99.0, 0)]
vol = B.volume.values; vr = pd.Series(vol).rolling(10).mean().values / pd.Series(vol).rolling(1440).mean().shift(1).values; pmp = (spk & np.nan_to_num(vr >= 3)).astype(np.bool_)
TS6 = np.r_[1e6, TSa, 8.0]; TH6 = np.r_[480, THa, 480]; byp = np.array([0, 0, 0, 0, 0, 1], np.bool_); F0 = np.zeros(n, bool)
def runR(Rsel, Rapp=None):
    E = []; z = np.zeros(n, np.bool_)
    for od, thr, hz, kL, kS, useB, cS, w, wf, thX, cLm in CF:
        L, S = selF(Rsel, thr, hz, od)
        if Rapp is not None: L, S = selF2(Rsel, Rapp, thr, hz, od)
        aL = L >= kL; aS = (S >= kS) & ~aL; eL, eS = gates(L, S, kL, kS, useB, cS, w, wf, 0.0, thX, cLm); E.append((aL, aS, eL, eS))
    X = st(*([(z, z, z, z)] + E + [(pmp, z, pmp, z)]))
    return z03.simN8(o, h, l, c, a, *X, TS6, TH6, spk, 60, 3, byp, 0.10, False, F0, F0)
def selF2(Rs, Ra, thr, hz, od):   # seçim tam veriyle (Rs), uygulama TV'de görünen hücre (Ra)
    nc = int(max(Rs.max(), Ra.max())) + 1; L = np.zeros(n, np.float32); S = np.zeros(n, np.float32); f = F[hz]; Rt = Rs[:mid]; ft = f[:mid]; fv = ~np.isnan(ft)
    for k in od:
        vt = V[k][:mid]; va = V[k]
        for side in (1, -1):
            m = vt == side; ca = np.bincount(Rt[m], minlength=nc); mv = m & fv
            sm = np.bincount(Rt[mv], weights=ft[mv], minlength=nc); cv = np.bincount(Rt[mv], minlength=nc)
            ok = (ca >= 400) & (cv > 0) & (sm / np.maximum(cv, 1) * side >= thr); (L if side > 0 else S)[:] += (va == side) & ok[Ra]
    return L, S
# portföy akış çukurları (10 dk MTM; seçimde KULLANILMAZ)
Pp = pd.read_pickle('/home/claude/port/portfoy_canli_10dk_utc.pkl'); Pp = Pp[Pp.index >= '2023-05-08']; pv = Pp.values.astype(float); pt = Pp.index
legs = []; mode = 0; hi = pv[0]; hi_i = 0; lo = pv[0]; lo_i = 0
for i in range(1, len(pv)):
    if mode >= 0:
        if pv[i] > hi: hi, hi_i = pv[i], i
        if hi - pv[i] >= 40: mode = -1; lo, lo_i = pv[i], i
    if mode == -1:
        if pv[i] < lo: lo, lo_i = pv[i], i
        if pv[i] - lo >= 40: legs.append((hi_i, lo_i, hi - lo)); mode = 1; hi, hi_i = pv[i], i
if mode == -1: legs.append((hi_i, lo_i, hi - lo))
LG = np.array([[np.searchsorted(idx.values, pt[a_].to_datetime64()), min(np.searchsorted(idx.values, pt[b_].to_datetime64()), n - 1), d_] for a_, b_, d_ in legs])
end_t = min(idx[-1], pt[-1]); w300 = np.datetime64(end_t - pd.Timedelta(days=300)); w90 = np.datetime64(end_t - pd.Timedelta(days=90))
pd_ = Pp.resample('1D').last().diff().dropna()
def portm(eq, since):
    s = LG[idx.values[LG[:, 0].astype(int)] >= since]; z = eq[s[:, 1].astype(int)] - eq[s[:, 0].astype(int)]
    zd = pd.Series(eq, idx).resample('1D').last().diff().reindex(pd_.index); pdd = pd_[pd_.index >= pd.Timestamp(since)]; zz = zd.reindex(pdd.index)
    kötü = pdd <= pdd.quantile(0.2)
    return z.sum(), (z > 0).mean() * 100 if len(z) else np.nan, z.sum() / s[:, 2].sum() * 100 if len(z) else np.nan, zz.corr(pdd), zz[kötü].sum()
out = []
def ad(nm, eq, T):
    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum()
    i300 = np.searchsorted(idx.values, w300); i90 = np.searchsorted(idx.values, w90); y3 = eq[i300::30] - eq[i300]; t3 = np.arange(len(y3))
    yb = np.r_[0, eq[np.searchsorted(idx.values, pd.date_range('2022-01-01', '2026-01-01', freq='YS').values)], eq[-1]]
    r = dict(model=nm, n=len(T), net=eq[-1], PF=gp / gl, MaxDD=dd.max(), DDp99=np.percentile(dd, 99), IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], TVpen=eq[-1] - eq[t23],
             son300=eq[-1] - eq[i300], R2_300=np.corrcoef(t3, y3)[0, 1] ** 2, son90=eq[-1] - eq[i90], yıllar=' '.join(f'{v:.0f}' for v in np.diff(yb)))
    for lab, since in (('tüm', np.datetime64('2023-05-08')), ('300', w300), ('90', w90)):
        cn, ck, kp, kr, kg = portm(eq, since); r.update({f'çuk{lab}$': cn, f'çuk{lab}%poz': ck, f'çuk{lab}%kapa': kp, f'kor{lab}': kr, f'kötüGün{lab}$': kg})
    out.append(r); print(pd.DataFrame([r]).round(2).to_string(index=False, header=len(out) == 1), flush=True)
print('portföy çukuru:', len(LG), '· son 300 g:', (idx.values[LG[:, 0].astype(int)] >= w300).sum(), '· son 90 g:', (idx.values[LG[:, 0].astype(int)] >= w90).sum(), flush=True)
eq, T = runR(R0i); ad('V2.7 baz', eq, T); np.save('/home/claude/zecwork/eq_V27.npy', eq)
for nm, ft_ in FEAT.items():
    b = bitn(ft_); Rn = R0i * 2 + b; eq, T = runR(Rn); ad(f'+{nm}', eq, T); np.save(f'/home/claude/zecwork/eq_LS_{nm}.npy', eq)
    if nm.startswith('LS'):
        b2 = b.copy(); b2[TVmask] = 1; eq, T = runR(Rn, R0i * 2 + b2); ad(f'+{nm} (TV-uyumlu: 2026-04 öncesi yok)', eq, T)
for n1, n2 in (('OIbirikim', 'LSglb'), ('OIfiyatUyum', 'LSglb'), ('OIbirikim', 'LStop')):
    Rn = (R0i * 2 + bitn(FEAT[n1])) * 2 + bitn(FEAT[n2]); eq, T = runR(Rn); ad(f'+{n1}+{n2}', eq, T)
Rn = R0i // 2 * 2 + bitn(FEAT['OIbirikim']); eq, T = runR(Rn); ad('XMR biti yerine OIbirikim', eq, T)
pd.DataFrame(out).to_csv('/home/claude/pine_hello_world/ZEC/research/OI_LS_BLOK_1DK.csv', index=False)
