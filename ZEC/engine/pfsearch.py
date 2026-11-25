# PORTFÖY KATKISI ODAKLI ZEC: V2.9 (V2.8 gölge + RSI10 ters) + R1 XMR zayıflık motoru + R2 alt risk-off bekçisi
# Amaç: 4.600 portföy + 200 $ ZEC → ΔSC4 ≥ 0, ΔCalmar ≥ 0, ΔMaxDD% ≤ 0, ΔP99% ≤ 0; seçim 2023-05 → 2025-02, doğrulama 2025-02 → 2026-11
import sys, glob, time, itertools, numpy as np, pandas as pd, numba
sys.path.insert(0, '/home/claude/zecwork'); from sc4 import sc4
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'tek'
Z = np.load('/home/claude/zecwork/base28.npz'); o, h, l, c, a = Z['o'], Z['h'], Z['l'], Z['c'], Z['a']; posA = Z['posA'].astype(np.int8); tms = Z['tms']; n = len(c); TB = Z['T']; TICK = 0.01
bEp = np.zeros(n); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
for r in TB:
    ei, xi = int(r[0]), int(r[1]); bEp[ei:xi] = r[3]; bXp[xi] = r[4]; bXd[xi] = int(r[2])
S = pd.Series; ix = pd.to_datetime(tms, unit='ms')
def rsi(x, N):
    d = np.diff(x, prepend=x[0]); u = S(np.maximum(d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy(); v = S(np.maximum(-d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy()
    return 100 - 100 / (1 + u / np.where(v == 0, 1e-12, v))
R10 = np.nan_to_num(rsi(c, 10), nan=50.0)
xmr = S(Z['xmr']).ffill().to_numpy(); btc = S(Z['btc']).ffill().to_numpy()
pe = []
for f in sorted(glob.glob('/home/claude/peer/1000PEPEUSDT_1m_FULL_*.csv.gz')) + sorted(glob.glob('/home/claude/peer/R_1000PEPEUSDT_*')):
    d = pd.read_csv(f); tc = 'open_time' if 'open_time' in d else ('open_time_ms' if 'open_time_ms' in d else 'time'); pe.append(S(d.close.values, d[tc].astype('int64').values))
pe = pd.concat(pe); pe = pe[~pe.index.duplicated(keep='last')].sort_index(); pep = pe.reindex(tms, method='ffill').to_numpy()
def ret(x, k): return np.r_[np.full(k, np.nan), x[k:] / x[:-k] - 1] * 100
def zret(x, k): r_ = ret(x, k); return r_ / S(r_).rolling(43200, min_periods=10000).std().to_numpy()
X24, B24 = ret(xmr, 1440), ret(btc, 1440); XZ = {k: zret(xmr, k) for k in (60, 240, 720)}; PZ = {k: zret(pep, k) for k in (60, 240, 720)}; ZR60 = ret(c, 60)
@numba.njit(cache=True)
def ov5(o, h, l, c, posA, bEp, bXp, bXd, R10, cond1, act1, H1, sl1, zweak, guard, act2, tick):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; rStp = 0.0
    r1Last = -10 ** 9; r1Stop = False; r1Own = False; r1Stp = 0.0; eq = np.zeros(n)
    for i in range(1, n):
        if pos != 0:
            xp = -1.0
            if r1Own:
                if h[i] >= r1Stp: xp = max(o[i], r1Stp) + tick; r1Stop = True; r1Own = False
            elif mode == 2 and pos == -tSide:
                if (pos == 1 and l[i] <= rStp) or (pos == -1 and h[i] >= rStp):
                    xp = (min(o[i], rStp) - tick) if pos == 1 else (max(o[i], rStp) + tick); mode = 0
            elif not np.isnan(bXp[i]) and pos == bXd[i]:
                xp = bXp[i]
            if xp > 0:
                real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
        D = posA[i]; want = D
        if D != prevD: mode = 0
        elif D != 0:
            if mode == 0:
                if (D > 0 and R10[i] >= 94) or (D < 0 and R10[i] <= 6):
                    want = -D; mode = 2; tPx = c[i]; tBar = i; tSide = D
            else:
                bi = (R10[i] <= 70 or i - tBar >= 240 or c[i] >= tPx * 1.03) if tSide > 0 else (R10[i] >= 30 or i - tBar >= 240 or c[i] <= tPx * 0.97)
                if bi: mode = 0
                else: want = -tSide if pos == -tSide else 0
        prevD = D
        if cond1[i]:
            if not r1Stop: r1Last = i
        else:
            r1Stop = False
        on = act1 > 0 and i - r1Last <= H1 and not r1Stop
        if on:
            if act1 == 1: want = 0 if want == 1 else want
            elif act1 == 2: want = -1
            else: want = -1 if zweak[i] else (0 if want == 1 else want)
        if act2 > 0 and guard[i]:
            if want == 1 and pos != 1: want = 0
            if act2 == 2 and pos == 1: want = 0
        if want != pos:
            if pos != 0:
                xp = c[i] - tick * pos; real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0; r1Own = False
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]
                if on and want == -1 and act1 >= 2: r1Own = True; r1Stp = ep * (1.0 + sl1)
                elif mode == 2 and want == -tSide: rStp = ep * (1.0 - want * 0.02)
        elif on and pos == -1 and act1 >= 2 and not r1Own:
            r1Own = True; r1Stp = ep * (1.0 + sl1)
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq
# ── portföy ──
Pp = pd.read_pickle('/home/claude/port/portfoy_canli_10dk_utc.pkl'); Pp = Pp[~Pp.index.duplicated()]
WIN = {'SEÇİM': (pd.Timestamp('2023-05-08'), pd.Timestamp('2025-02-01')), 'DOĞRULAMA': (pd.Timestamp('2025-02-01'), Pp.index[-1])}
def legs_of(p):
    pv = p.values; L = []; mode = 0; hi = pv[0]; hi_i = 0; lo = pv[0]; lo_i = 0
    for i in range(1, len(pv)):
        if mode >= 0:
            if pv[i] > hi: hi, hi_i = pv[i], i
            if hi - pv[i] >= 40: mode = -1; lo, lo_i = pv[i], i
        if mode == -1:
            if pv[i] < lo: lo, lo_i = pv[i], i
            if pv[i] - lo >= 40: L.append((p.index[hi_i], p.index[lo_i], hi - lo)); mode = 1; hi, hi_i = pv[i], i
    if mode == -1: L.append((p.index[hi_i], p.index[lo_i], hi - lo))
    return L
LEGS = legs_of(Pp); PM = Pp.resample('ME').last().diff().dropna()
BASE = {}
for w, (a0, a1) in WIN.items():
    p = Pp[(Pp.index >= a0) & (Pp.index < a1)]; p = p - p.iloc[0]; BASE[w] = sc4(p, 4600, True)
def pmetrics(eqZ, lab):
    z = S(eqZ, ix).reindex(Pp.index, method='ffill').fillna(0.0) * 0.4          # 200 $ nominal
    r = dict(model=lab)
    for w, (a0, a1) in WIN.items():
        m = (Pp.index >= a0) & (Pp.index < a1); p = Pp[m] - Pp[m].iloc[0]; zz = z[m] - z[m].iloc[0]; s_ = sc4(p + zz, 4800, True); b = BASE[w]
        r[f'{w} ΔSC4'] = s_['SC4'] - b['SC4']; r[f'{w} ΔCalmar'] = s_['Calmar'] - b['Calmar']; r[f'{w} ΔMaxDD%'] = s_['MaxDD'] - b['MaxDD']; r[f'{w} ΔP99%'] = s_['P99'] - b['P99']
        lg = [(x, y, d) for x, y, d in LEGS if a0 <= x < a1]; g = np.array([z.asof(y) - z.asof(x) for x, y, d in lg]) / 0.4
        r[f'{w} çukur$'] = g.sum(); r[f'{w} çukur%+'] = (g > 0).mean() * 100
        pm = PM[(PM.index >= a0) & (PM.index < a1)]; zm = (z / 0.4).resample('ME').last().diff().reindex(pm.index); weak = pm <= pm.quantile(0.25)
        r[f'{w} zayıfAy+'] = f'{int((zm[weak] > 0).sum())}/{int(weak.sum())}'; r[f'{w} korAy'] = zm.corr(pm)
    e30 = eqZ[::30]; dd = np.maximum.accumulate(e30) - e30; hr = np.diff(eqZ[::60])
    r.update(net=eqZ[-1], MaxDD=dd.max(), MaxDDnom=dd.max() / 5, std1s=hr.std(), son300=eqZ[-1] - eqZ[np.searchsorted(tms, tms[-1] - 300 * 86400000)])
    return r
F0 = np.zeros(n, np.bool_)
def run(c1, a1, H1, sl1, g2, a2, lab):
    return pmetrics(ov5(o, h, l, c, posA, bEp, bXp, bXd, R10, F0 if c1 is None else c1, a1, H1, sl1, ZR60 < 0, F0 if g2 is None else g2, a2, TICK), lab)
C1 = {}
for th, k, m in itertools.product((0.0, 2.0, 4.0), (60, 240), (1.5, 2.0, 2.5)):
    C1[f'XMR−BTC 24s ≤ −%{th:g} & XMR {k}dk z ≤ −{m}'] = np.nan_to_num((X24 - B24 <= -th) & (XZ[k] <= -m)).astype(np.bool_)
G2 = {}
for k, zt in itertools.product((60, 240, 720), (1.0, 1.5, 2.0)):
    G2[f'PEPE & XMR {k}dk z ≤ −{zt}'] = np.nan_to_num((PZ[k] <= -zt) & (XZ[k] <= -zt)).astype(np.bool_)
rows = []; t0 = time.time()
if STAGE == 'tek':
    rows.append(run(None, 0, 0, 0.0, None, 0, 'V2.9 (kural yok)')); print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)
    for cn, cnd in C1.items():
        for a1, H1, sl1 in ((1, 60, 0.0), (1, 240, 0.0), (2, 60, 0.02), (2, 60, 0.04), (2, 240, 0.02), (2, 240, 0.04), (3, 60, 0.02), (3, 240, 0.02), (3, 240, 0.04)):
            rows.append(run(cnd, a1, H1, sl1, None, 0, f'R1 {cn} | eylem{a1} tut{H1} sl%{sl1 * 100:g}'))
    for gn, g in G2.items():
        for a2 in (1, 2): rows.append(run(None, 0, 0, 0.0, g, a2, f'R2 {gn} | ' + ('yeni long yok' if a2 == 1 else 'long kapat + yeni yok')))
    pd.DataFrame(rows).to_pickle('/home/claude/zecwork/pf_tek.pkl'); print('bitti', len(rows), f'{time.time() - t0:.0f} sn')
