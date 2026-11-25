exec(open('/home/claude/zecwork/mega5.py').read().split("rows = []; t0 = time.time()")[0])
import importlib; importlib.reload(z03)
tv = np.load('/home/claude/zecwork/tv_V.npz')
for k in tv.files:
    if k not in V: V[k] = tv[k].astype(np.float32)
PR = ['P09_spike', 'P07_rsi', 'P08_zema', 'P10_bb', 'P15_resid', 'P04_htf4h']; OI = ['P12_oiLiq', 'P11_oiCont']; EXT = ['P01_ema', 'P02_st', 'P03_htf30', 'P05_don', 'P06_roc', 'P13_btc', 'P14_xmr', 'X16_flow']
t23 = p('2023-02-06'); R0 = RL[0]; LS = {}
def LSget(sn, od, thr, hz):
    global order
    if (sn, thr, hz) not in LS: order = od; LS[(sn, thr, hz)] = select(R0, thr, hz)
    return LS[(sn, thr, hz)]
def sig(sn, od, thr, hz, kL, kS, useB, cS, w, wf, thX, cLm):
    L, S = LSget(sn, od, thr, hz); aL = L >= kL; aS = (S >= kS) & ~aL; eL, eS = gates(L, S, kL, kS, useB, cS, w, wf, 0.0, thX, cLm); return aL, aS, eL, eS
B1 = sig('B', PR + OI + ['X17_basis'], 25, 480, 2, 2, 1, 0, 10, 0.4, 4.0, 2); S1 = sig('A', PR + OI, 10, 480 // 4, 2, 1, 0, 0, 10, 0.4, 99.0, 0)
def met(eq, T):
    x30 = eq[::30]; dd = np.maximum.accumulate(x30) - x30; w_ = eq[t23:] - eq[t23]
    return dict(n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=dd.max(), DDp99=np.percentile(dd, 99), TVpencere=w_[-1],
                m0=T[T[:, 7] == 0, 5].sum(), m1=T[T[:, 7] == 1, 5].sum(), m2=T[T[:, 7] == 2, 5].sum(), n2=(T[:, 7] == 2).sum())
rows = []; import gc as _gc
st = lambda *E: tuple(np.ascontiguousarray(np.stack([e[j] for e in E])) for j in range(4))
AL, AS, EL, ES = st(B1, S1); eq, T = z03.simN(o, h, l, c, a, AL, AS, EL, ES, np.array([28.0, 28.0]), np.array([3840, 480]), spk, 60); rows.append(dict(yığın='B1 + S1', **met(eq, T)))
for sn, od in (('A', PR + OI), ('G', PR + OI + EXT), ('B', PR + OI + ['X17_basis'])):
    for thr, hz in itertools.product((10, 15, 25), (120, 240, 480)):
        order = od; L3, S3 = select(R0, thr, hz)
        for kL, kS, wf in itertools.product((1, 2), (1, 2), (0.0, 0.4)):
            aL = L3 >= kL; aS = (S3 >= kS) & ~aL; eL, eS = gates(L3, S3, kL, kS, 0, 0, 10, wf, 0.0, 99.0, 0)
            AL, AS, EL, ES = st(B1, S1, (aL, aS, eL, eS))
            for ts3, tH3 in itertools.product((16.0, 28.0), (480, 960)):
                eq, T = z03.simN(o, h, l, c, a, AL, AS, EL, ES, np.array([28.0, 28.0, ts3]), np.array([3840, 480, tH3]), spk, 60)
                rows.append(dict(yığın=f'B1 + S1 + {sn} thr{thr} hz{hz} k{kL}/{kS} fitil{wf} ts{ts3:.0f} tH{tH3}', **met(eq, T)))
            del AL, AS, EL, ES, aL, aS, eL, eS; _gc.collect()
        print(sn, thr, hz, len(rows), flush=True)
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/ORKESTRA3_1DK.csv', index=False); pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 60)
print(R.iloc[:1].round(0).to_string(index=False)); f = R[(R.IS > 0) & (R.OOS > 0)]
print(f.sort_values('net', ascending=False).head(10).round(0).to_string(index=False)); print('\nMaxDD ≤ 400:'); print(f[f.MaxDD <= 400].sort_values('net', ascending=False).head(6).round(0).to_string(index=False))
