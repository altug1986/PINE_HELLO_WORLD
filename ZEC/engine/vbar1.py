# (1) ETH 4 s ADX VETO (ters girişi kes)  (2) BASKI/ULAŞAMAMA: 6 bar içinde yön mesafesine ulaşamazsa ters yöne geç
exec(open('/home/claude/zecwork/eth1.py').read().split("eq, T = runC(CF); ad('V2.7 baz', eq, T)")[0])
importlib.reload(z03)
def buildX(CFx):
    E_ = []; z = np.zeros(n, np.bool_)
    for od, thr, hz, kL, kS, useB, cS, w, wf, thX, cLm in CFx:
        L, S = selF(R0i, thr, hz, od); aL = L >= kL; aS = (S >= kS) & ~aL; eL, eS = gates(L, S, kL, kS, useB, cS, w, wf, 0.0, thX, cLm); E_.append((aL, aS, eL, eS))
    return [np.array(x) for x in st(*([(z, z, z, z)] + E_ + [(pmp, z, pmp, z)]))]
X0 = buildX(CF)
def sim(X, th0=480, fN=0, fM=0.0, fR=False):
    return z03.simN9(o, h, l, c, a, *X, TS6, np.r_[th0, TH6[1:]], spk, 60, 3, byp, 0.10, False, F0, F0, fN, fM, fR)
eq, T = sim(X0); ad('V2.7 baz', eq, T)
ethTr = {}
for thA in (20, 25, 30):
    ethTr[thA] = np.nan_to_num(map1m(pd.Series(np.where(ax >= thA, np.sign(pdi - mdi), 0.0), E4.index), 4))
    X = [x.copy() for x in X0]; X[2][1:5] &= ~(ethTr[thA] == -1); X[3][1:5] &= ~(ethTr[thA] == 1)
    eq, T = sim(X); ad(f'ETH 4s ADX≥{thA} veto', eq, T)
for fN in (3, 6, 10):
    for fM in (0.5, 1.0, 2.0):
        for fR, th0 in ((False, 480), (True, 60), (True, 240)):
            if fN != 6 and (fM != 1.0 or th0 == 60): continue
            eq, T = sim(X0, th0, fN, fM, fR); ad(f'baskı {fN} bar {fM}×ATR ' + (f'TERS tut{th0}' if fR else 'yalnız kapat'), eq, T)
pd.DataFrame(out).to_csv('/home/claude/pine_hello_world/ZEC/research/ETH_VETO_BASKI_1DK.csv', index=False)
