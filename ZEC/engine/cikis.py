# NEDEN LONG'TAN ÇIKIYOR? Orkestra çıkış nedenleri (stop / üst-eş motor tek bar ters oy / sahip motor süre doldu) + düzeltmeler: ters oy teyidi, ortak ters yön, kârdayken süre çıkışı yok
exec(open('/home/claude/zecwork/akilli.py').read().split("rows = []; P = {}")[0])
from numba import njit
@njit(cache=False)
def simX(o, h, l, c, atr, AL, AS, EL, ES, TS, TH, spk, blkN, mPers, byp, cap, oppN, oppCons, keepL, keepS):
    K, n = AL.shape; eq = np.zeros(n); real = 0.0; pos = 0; own = -1; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; noS = -1; ae = 0.0; oc = 0
    T = np.zeros((n // 2 + 10, 8)); nt = 0; w = np.zeros(K, np.int64); runD = 0; runN = 0
    for i in range(1, n):
        if pos != 0 and ((pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop)):
            xp = (min(o[i], stop) - 0.01) if pos == 1 else (max(o[i], stop) + 0.01); pnl = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = 1; T[nt, 7] = own; nt += 1; pos = 0; own = -1; oc = 0
        if spk[i]: noS = i + blkN
        for k in range(K): w[k] = 1 if AL[k, i] else (-1 if (AS[k, i] and i > noS) else 0)
        aL_ = False; aS_ = False
        for kk in range(K):
            if w[kk] == 1: aL_ = True
            elif w[kk] == -1: aS_ = True
        Dn = 0 if (aL_ and aS_) else (1 if aL_ else (-1 if aS_ else 0))
        if Dn != 0 and Dn == runD: runN += 1
        else: runD = Dn; runN = 1 if Dn != 0 else 0
        guard = pos == -1 and spk[i]
        if pos != 0 and not guard:
            for j in range(own):
                if w[j] == pos: own = j; ti = i; stop = ep - pos * min(TS[j] * ae, cap * ep); break
        ex = False; blk = 0; why = 0
        if pos != 0 and not guard and i - ti >= TH[own]:
            for kk in range(K):
                if w[kk] == pos: own = kk; ti = i; stop = ep - pos * min(TS[kk] * ae, cap * ep); break
        if pos != 0:
            tout = i - ti >= TH[own]
            if tout and ((pos == 1 and keepL) or (pos == -1 and keepS)) and (c[i] - ep) * pos > 0: tout = False
            opp = False
            for j in range(own + 1):
                if w[j] != 0 and w[j] != pos: opp = True
            if oppCons and opp and Dn != -pos: opp = False
            oc = oc + 1 if opp else 0
            opp = oc >= oppN
            ex = guard or opp or tout; why = 4 if guard else (2 if opp else 3)
            blk = pos if (tout and w[own] == pos) else 0
        if ex:
            xp = c[i] - 0.01 * pos; pnl = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = why; T[nt, 7] = own; nt += 1; pos = 0; own = -1; oc = 0
        if pos != 0 and w[own] == pos: ti = i
        if pos == 0 and not np.isnan(atr[i]):
            for k in range(K):
                ok = byp[k] or (runD == w[k] and runN >= mPers)
                if ok and w[k] != 0 and w[k] != blk and ((w[k] == 1 and EL[k, i]) or (w[k] == -1 and ES[k, i])):
                    pos = w[k]; own = k; ep = c[i] + 0.01 * pos; q = 500.0 / ep; ei = i; ti = i; ae = atr[i]; stop = ep - pos * min(TS[k] * ae, cap * ep); oc = 0; break
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]
pc = pmp & np.nan_to_num(btc / np.r_[np.full(10, np.nan), btc[:-10]] - 1 >= 0.005, nan=False); XV = [x.copy() for x in XB]; XV[0][5] = pc; XV[2][5] = pc
THv = np.r_[480, TH6[1:]].astype(np.int64); args = (np.asarray(o, float), np.asarray(h, float), np.asarray(l, float), cc, np.asarray(a, float))
def chainX(oppN, cons, kL, kS):
    eqb, Tb = simX(*args, *XV, TS6, THv, spk, 60, 3, byp, 0.10, oppN, cons, kL, kS)
    P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
    for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
    vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR = np.zeros((20000, 4))
    run29(args[0], args[1], args[2], cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR)
    e, nt, _ = f3E(args[0], args[1], args[2], cc, 0.01, vp, vx, vn, np.ones(n, np.bool_), EVT, 1440, 0.08, np.zeros(n, np.int64), 0, 0.0)
    return e, Tb
e0, T0_ = chainX(1, False, False, False); print(f'V3.2 yeniden üretim {e0[-1]:.1f} (10.316,5)')
# çıkış nedenleri (orkestra gölge pozisyonu) ve çıkış sonrası fiyat (tutulsaydı)
def fw(H): r = np.full(n, np.nan); r[:-H] = cc[H:] / cc[:-H] - 1; return r
F4, F12 = fw(240), fw(720); xi = T0_[:, 1].astype(int); dv = T0_[:, 2]; ad = {1: 'stop', 2: 'üst/eş motor ters oy', 3: 'sahip motor süresi doldu', 4: 'pompa koruması'}
S_ = pd.DataFrame({'yön': np.where(dv > 0, 'long', 'short'), 'neden': [ad[int(x)] for x in T0_[:, 6]], 'pnl': T0_[:, 5], 'sonra4s': F4[xi] * dv * 1e4, 'sonra12s': F12[xi] * dv * 1e4, 'yarı': np.where(xi < midI, 1, 2)})
print(S_.groupby(['yön', 'neden']).agg(adet=('pnl', 'size'), net=('pnl', 'sum'), sonra4s_bps=('sonra4s', 'mean'), sonra12s_bps=('sonra12s', 'mean')).round(1).to_string())
rows = [met(e0, 'V3.2 (şimdiki çıkış)')]
for oppN, cons, kL, kS in ((2, False, False, False), (3, False, False, False), (5, False, False, False), (10, False, False, False), (1, True, False, False), (3, True, False, False),
                           (1, False, True, False), (1, False, True, True), (3, False, True, False), (3, True, True, False)):
    e, _ = chainX(oppN, cons, kL, kS); rows.append(met(e, f'ters oy {oppN} bar teyit' + (' + ortak ters yön' if cons else '') + (' + kârdaki LONG süreyle çıkmaz' if kL else '') + (' + kârdaki SHORT da' if kS else '')))
R = pd.DataFrame(rows); b = R.iloc[0]
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - b[k]
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 70); print(R.round(2).to_string(index=False)); R.to_csv('/home/claude/zecwork/cikis.csv', index=False); S_.to_csv('/home/claude/zecwork/cikis_neden.csv', index=False)
# 23 Kas son long'un çıkış nedeni
k = np.flatnonzero((T0_[:, 0] >= i23) & (dv > 0))
for r in T0_[k]: print('23 Kas long', pd.to_datetime(tms[int(r[0])], unit='ms') + pd.Timedelta('3h'), '→', pd.to_datetime(tms[int(r[1])], unit='ms') + pd.Timedelta('3h'), f'{r[5]:+.1f} $', ad[int(r[6])], 'sahip', int(r[7]))
