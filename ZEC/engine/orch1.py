# ORKESTRA: ana motor (V1.1 / B1) boşken yedek motor işlem açar — aynı 500 $ birim, TV'de hesaplanabilen veriler
exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
import importlib; importlib.reload(z03)
tv = np.load('/home/claude/zecwork/tv_V.npz')
for k in tv.files:
    if k not in V: V[k] = tv[k].astype(np.float32)
PR = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h']; OI = ['P12_oiLiq', 'P11_oiCont']; EXT = ['P01_ema', 'P02_st', 'P03_htf30', 'P05_don', 'P06_roc', 'P13_btc', 'P14_xmr', 'X16_flow']
t23 = p('2023-02-06'); R0 = RL[0]
def sig(od, thr, hz, kL, kS, useB, cS, w, wf, thX, cLm):
    global order; order = od; L, S = select(R0, thr, hz); aL = L >= kL; aS = (S >= kS) & ~aL; eL, eS = gates(L, S, kL, kS, useB, cS, w, wf, 0.0, thX, cLm); return aL, aS, eL, eS
PRIM = {'V1.1': (sig(PR + OI, 25, 480, 2, 2, 1, 0, 10, 0.4, 4.0, 2), 40.0, 1920), 'B1': (sig(PR + OI + ['X17_basis'], 25, 480, 2, 2, 1, 0, 10, 0.4, 4.0, 2), 28.0, 3840)}
SEC = {}
for nm, od in (('aynı set', PR + OI), ('geniş set', PR + OI + EXT)):
    for thr, hz in itertools.product((10, 15), (120, 240)):
        for kL, kS, wf in itertools.product((1, 2), (1, 2), (0.0, 0.4)):
            SEC[(nm, thr, hz, kL, kS, wf)] = sig(od, thr, hz, kL, kS, 0, 0, 10, wf, 99.0, 0)
print('yedek sinyal setleri:', len(SEC), flush=True)
rows = []
def met(eq, T):
    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; w = eq[t23:] - eq[t23]; o2 = T[:, 7] == 2
    return dict(n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp99=np.percentile(dd, 99), TVpencere=w[-1], yedekİşlem=o2.sum(), yedekNet=T[o2, 5].sum(), short=T[T[:, 2] < 0, 5].sum())
for pn, (ps, ts1, tH1) in PRIM.items():
    eq, T = z03.sim11(o, h, l, c, a, ps[0], ps[1], ps[2], ps[3], ts1, tH1, spk, 60); rows.append(dict(ana=pn, yedek='—', **met(eq, np.c_[T[:, :7], np.ones(len(T))])))
    for key, ss in SEC.items():
        for ts2, tH2 in itertools.product((16.0, 28.0), (480, 960)):
            eq, T = z03.sim12(o, h, l, c, a, *ps, ts1, tH1, *ss, ts2, tH2, spk, 60)
            rows.append(dict(ana=pn, yedek=f'{key[0]} thr{key[1]} hz{key[2]} k{key[3]}/{key[4]} fitil{key[5]} ts{ts2:.0f} tH{tH2}', **met(eq, T)))
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/ORKESTRA_1DK.csv', index=False); pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 70)
print(R[R.yedek == '—'].round(0).to_string(index=False))
f = R[(R.IS > 0) & (R.OOS > 0)]; print('\nen yüksek net:'); print(f.sort_values('net', ascending=False).head(12).round(0).to_string(index=False))
print('\nMaxDD ≤ 450:'); print(f[f.MaxDD <= 450].sort_values('net', ascending=False).head(8).round(0).to_string(index=False))
