# Portföy çukuru ölçütü + TV sürümü (fiyat + OI) son seçimi
exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
Pp = pd.read_pickle('/home/claude/port/portfoy_canli_10dk_utc.pkl'); Pp = Pp['2023-05-08':]
v = Pp.values; tt = Pp.index; legs = []; mode = 0; hi = v[0]; hi_i = 0; lo = v[0]; lo_i = 0
for i in range(1, len(v)):                       # zigzag, 40 $ dönüş eşiği: zirve → dip bacakları
    if mode >= 0:
        if v[i] > hi: hi, hi_i = v[i], i
        if hi - v[i] >= 40: mode = -1; lo, lo_i = v[i], i
    if mode == -1:
        if v[i] < lo: lo, lo_i = v[i], i
        if v[i] - lo >= 40: legs.append((hi_i, lo_i, hi - lo)); mode = 1; hi, hi_i = v[i], i
if mode == -1: legs.append((hi_i, lo_i, hi - lo))
W = [(tt[a_], tt[b_], d_) for a_, b_, d_ in legs if d_ > 40]; print('çukur sayısı:', len(W), '· toplam portföy kaybı:', round(sum(d for *_, d in W)), flush=True)
wi = np.array([[np.searchsorted(idx.values, a_.to_datetime64()), min(np.searchsorted(idx.values, b_.to_datetime64()), n - 1)] for a_, b_, _ in W]); wl = np.array([d for *_, d in W])
half = pd.Timestamp('2025-02-15')
def cuk(eq):
    z = eq[wi[:, 1]] - eq[wi[:, 0]]; f1 = np.array([a_ < half for a_, _, _ in W])
    return dict(çukurNet=z.sum(), çukurKâr=(z > 0).mean() * 100, kapama=z.sum() / wl.sum() * 100, çukurNet_ilk=z[f1].sum(), çukurNet_son=z[~f1].sum())
PR = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h']; OI = ['P12_oiLiq', 'P11_oiCont']
rows = []
for nm, od in (('V1.0 (fiyat)', PR), ('fiyat+OI', PR + OI)):
    order = od; sel.cache_clear()
    for thr, hz in itertools.product((15, 25), (240, 480, 960)):
        for kL, kS, ts, tH in itertools.product((1, 2, 3), (1, 2, 3), (28.0, 40.0), (1920, 3840)):
            if nm.startswith('V1.0') and (thr, hz, kL, kS, ts, tH) != (25, 480, 2, 2, 28.0, 3840): continue
            eq, T = run(0, thr, hz, kL, kS, 1, 0, 10, 0.4, 0.0, ts, tH, 4.0, 2)
            if len(T) < 300: continue
            x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; d_ = np.diff(x30[::48]); gp = T[T[:, 5] > 0, 5].sum(); gl = -T[T[:, 5] < 0, 5].sum()
            rows.append(dict(set=nm, thr=thr, hz=hz, kL=kL, kS=kS, ts=ts, tH=tH, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], PF=gp / gl, MaxDD=dd.max(), DDp99=np.percentile(dd, 99),
                             Sharpe=d_.mean() / d_.std() * np.sqrt(365), Kasım=eq[-1] - eq[p('2026-11-01')], **cuk(eq)))
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/TV_CUKUR_SECIM_1DK.csv', index=False); pd.set_option('display.width', 260)
print(R[R.set.str.startswith('V1.0')].round(2).to_string(index=False))
f = R[(R.set == 'fiyat+OI') & (R.IS > 0) & (R.OOS > 0)]
print('\nfiyat+OI — çukur hedefi (net>0, ≥%50 kâr, ≥%30 kapama) tutan:', ((f.çukurNet > 0) & (f.çukurKâr >= 50) & (f.kapama >= 30)).sum(), '· çukur net>0:', (f.çukurNet > 0).sum())
print(f.sort_values('net', ascending=False).head(8).round(2).to_string(index=False))
print('\nçukur netine göre:'); print(f.sort_values('çukurNet', ascending=False).head(8).round(2).to_string(index=False))
