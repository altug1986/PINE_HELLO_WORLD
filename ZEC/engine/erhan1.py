# Video dersleri: (1) kayma+komisyon stresi (2) EN UZUN YATAY KALMA = seviyeyi kalıcı geçene kadar süre (3) tek vurgun bağımlılığı (4) en iyi yıllar çıkarılmış beklenti
exec(open('/home/claude/zecwork/hold1.py').read().split("rows = []")[0].replace("RGi = np.nan_to_num(RG).astype(np.int64); ", ""))
RGi = np.nan_to_num(RG).astype(np.int64); import ast
def flat(x):                                   # x: 30 dk eğri → en uzun "kalıcı geçememe" süresi (gün) ve şu anki
    sm = np.minimum.accumulate(x[::-1])[::-1]; s = np.searchsorted(sm, x, side='right'); d = s - np.arange(len(x)); j = int(np.argmax(d))
    return d.max() * 30 / 1440, idx[min(j * 30, n - 1)].date(), (len(x) - np.searchsorted(sm, x[-1], side='left')) * 30 / 1440
def run(cfg):
    thr, hz, kL, kS, mod, k4, ts, tH = cfg; L, S = select(thr, hz); aL = (L >= kL); aS = (S >= kS) & ~aL
    return z03.sim10(o, h, l, c, a, aL, aS, float(ts), int(tH), spk, 60, RGi, A4 * k4, int(mod))
def stress(eq, T):                              # +0,04 % × 2 bacak komisyon + 2 tick × 2 bacak ek kayma → toplam %0,08 + 3 tick
    q = 500 / T[:, 3]; xc = 0.0004 * q * (T[:, 3] + T[:, 4]) + 4 * 0.01 * q; cum = np.zeros(n); np.add.at(cum, T[:, 1].astype(int), xc); return eq - np.cumsum(cum)
C = {'R5 (BoB)': (15, 240, 2, 3, 0, 3, 16.0, 1920), 'HOLD-maks': (15, 240, 2, 4, 1, 8, 16.0, 1920), 'HOLD-dengeli': (10, 480, 2, 3, 2, 3, 24.0, 1920)}
ens = ast.literal_eval('[' + pd.read_csv('/home/claude/pine_hello_world/ZEC/research/ENS_1DK.csv').motorlar.iloc[-1].replace(' | ', ',').replace('np.float64(', '').replace('), ', ', ') + ']')
ens = [tuple(float(str(v).strip('()')) for v in e) for e in ens]
yr = idx.year.values; out = []
def evals(nm, eq, T, x30=None):
    r = {'sistem': nm}
    for tag, e in (('', eq), ('_stres', stress(eq, T) if T is not None else None)):
        if e is None: continue
        x = e[::30]; dd = np.maximum.accumulate(x) - x; f, fy, now = flat(x)
        r.update({'net' + tag: e[-1], 'MaxDD' + tag: dd.max(), 'DDp98' + tag: np.percentile(dd, 98), 'yatayGün' + tag: f, 'yatayBaş' + tag: fy, 'şuAnYatay' + tag: now})
    Y = pd.Series(eq[np.r_[np.flatnonzero(np.diff(yr)), n - 1]], index=np.unique(yr)).diff().fillna(eq[np.flatnonzero(np.diff(yr))[0]])
    r['yıl$'] = ' '.join(f'{k % 100}:{v:.0f}' for k, v in Y.items()); r['beklenti(en iyi 2 yıl hariç ort/yıl)'] = Y.sort_values().iloc[:-2].mean()
    if T is not None:
        tp = np.sort(T[:, 5]); r['n'] = len(T); r['top3%'] = tp[-3:].sum() / eq[-1] * 100; r['net−top3'] = eq[-1] - tp[-3:].sum()
    bp = np.diff(eq[blk]); r['net−enİyiBlok'] = eq[-1] - bp.max(); r['enİyiBlok'] = bp.max()
    for k_, (a_, b_) in WIN.items(): r[k_] = eq[b_] - eq[a_]
    out.append(r)
for nm, cfg in C.items():
    eq, T = run(cfg); evals(nm, eq, T)
EQ = []; TT = []
for cfg in ens:
    eq, T = run(cfg); EQ.append(eq / len(ens)); T = T.copy(); T[:, 5] /= len(ens); TT.append(T)
Te = np.concatenate(TT); eqE = np.sum(EQ, 0)
def stressE(eq, T):  # ensemble: maliyet de 1/K
    q = 500 / len(ens) / T[:, 3]; xc = 0.0004 * q * (T[:, 3] + T[:, 4]) + 4 * 0.01 * q; cum = np.zeros(n); np.add.at(cum, T[:, 1].astype(int), xc); return eq - np.cumsum(cum)
_s = stress; stress = stressE; evals('ENS-6 (aynı 500 bölünmüş)', eqE, Te); stress = _s
R = pd.DataFrame(out); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(R.T.to_string()); R.to_csv('/home/claude/pine_hello_world/ZEC/research/ERHAN_TEST_1DK.csv', index=False)
