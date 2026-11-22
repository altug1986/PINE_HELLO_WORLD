import numpy as np, pandas as pd, numba as nb
from functools import lru_cache
D = '/home/claude/data/'
TICK, COMM, CASH, CAP = 1e-7, 0.0004, 500.0, 10000.0
SPLIT = pd.Timestamp('2025-05-20', tz='UTC')            # IS < SPLIT <= OOS

def load(tf):
    b = pd.read_parquet(D + f'pepe_{tf}.parquet')
    return b, b.index.as_unit('s').asi8

B5, T5 = load('5min')
H, L_, C = B5.high.values, B5.low.values, B5.close.values
N = len(C)

# ---------- TV-semantics indicators ----------
@nb.njit(cache=True)
def rma(x, n):
    out = np.full(x.size, np.nan); prev = np.nan; run = 0; a = 1.0 / n
    for i in range(x.size):
        v = x[i]
        if np.isnan(v):
            run = 0; out[i] = prev; continue
        run += 1
        if np.isnan(prev):
            if run >= n:
                s = 0.0
                for j in range(i - n + 1, i + 1): s += x[j]
                prev = s / n
        else:
            prev = a * v + (1 - a) * prev
        out[i] = prev
    return out

@nb.njit(cache=True)
def ema(x, n):
    out = np.full(x.size, np.nan); prev = np.nan; a = 2.0 / (n + 1)
    for i in range(x.size):
        if np.isnan(prev):
            if i >= n - 1:
                s = 0.0
                for j in range(i - n + 1, i + 1): s += x[j]
                prev = s / n
        else:
            prev = a * x[i] + (1 - a) * prev
        out[i] = prev
    return out

@nb.njit(cache=True)
def dmi(h, l, c, dl, al):
    n = h.size; tr = np.full(n, np.nan); pd_ = np.full(n, np.nan); md = np.full(n, np.nan)
    for i in range(1, n):
        up = h[i] - h[i-1]; dn = l[i-1] - l[i]
        pd_[i] = up if (up > dn and up > 0) else 0.0
        md[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    trur = rma(tr, dl); p = 100 * rma(pd_, dl) / trur; m = 100 * rma(md, dl) / trur
    lp = np.nan; lm = np.nan                                   # fixnan
    for i in range(n):
        if np.isnan(p[i]): p[i] = lp
        else: lp = p[i]
        if np.isnan(m[i]): m[i] = lm
        else: lm = m[i]
    s = p + m; x = np.abs(p - m) / np.where(s == 0, 1.0, s)
    return p, m, 100 * rma(x, al)

@nb.njit(cache=True)
def supertrend_dir(h, l, c, f, ap):                            # -1 = up (TV convention)
    n = h.size; tr = np.empty(n); tr[0] = h[0] - l[0]
    for i in range(1, n): tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr = rma(tr, ap); d = np.full(n, np.nan); ub = np.full(n, np.nan); lb = np.full(n, np.nan); st = np.full(n, np.nan)
    for i in range(n):
        src = (h[i] + l[i]) / 2; u = src + f * atr[i]; lo = src - f * atr[i]
        pl = lb[i-1] if i > 0 and not np.isnan(lb[i-1]) else 0.0
        pu = ub[i-1] if i > 0 and not np.isnan(ub[i-1]) else 0.0
        if i > 0 and not (lo > pl or c[i-1] < pl): lo = pl
        if i > 0 and not (u < pu or c[i-1] > pu): u = pu
        lb[i] = lo; ub[i] = u
        if i == 0 or np.isnan(atr[i-1]): dd = 1.0
        elif st[i-1] == pu: dd = -1.0 if c[i] > u else 1.0
        else: dd = 1.0 if c[i] < lo else -1.0
        d[i] = dd; st[i] = lo if dd == -1.0 else u
    return d

def htf_map(arr, th, tfsec):                                   # lookahead_on + [1] == last completed HTF bar
    b = (T5 // tfsec) * tfsec; idx = np.searchsorted(th, b)
    out = np.full(N, np.nan); ok = idx >= 1; out[ok] = arr[idx[ok] - 1]; return out

@lru_cache(None)
def htf(tf):
    b, th = load(f'{tf}min'); return b.high.values, b.low.values, b.close.values, th, tf * 60

@lru_cache(None)
def adx5(dl=14, al=14): return dmi(H, L_, C, dl, al)

@lru_cache(None)
def adx_htf(tf, dl=14, al=14):
    h, l, c, th, s = htf(tf); p, m, a = dmi(h, l, c, dl, al)
    return htf_map(a, th, s), htf_map(p - m, th, s)

@lru_cache(None)
def st_htf(tf, f, ap):
    if tf == 5: return supertrend_dir(H, L_, C, f, ap)
    h, l, c, th, s = htf(tf); return htf_map(supertrend_dir(h, l, c, f, ap), th, s)

@lru_cache(None)
def ema_htf(tf, n):
    if tf == 5: return ema(C, n)
    h, l, c, th, s = htf(tf); return htf_map(ema(c, n), th, s)

@lru_cache(None)
def mom(n):
    m = np.full(N, np.nan); m[n:] = (C[n:] / C[:-n] - 1) * 100; return m

@lru_cache(None)
def donch(n):
    hh = pd.Series(H).rolling(n).max().shift(1).values; ll = pd.Series(L_).rolling(n).min().shift(1).values
    return hh, ll

def consec(x, k):                                              # true k bars in a row
    if k <= 1: return x
    cs = np.concatenate([[0], np.cumsum(x.astype(np.int64))])
    r = np.zeros(N, bool); r[k-1:] = (cs[k:] - cs[:-k]) == k; return r

# ---------- SAR broker (process_orders_on_close, 1 tick slip, % comm both legs, fixed cash) ----------
@nb.njit(cache=True)
def run(c, Ls, Ss, hold, tick, comm, cash):
    n = c.size; pos = 0; ep = 0.0; q = 0.0; ei = -1; real = 0.0; k = 0
    eq = np.empty(n); te = np.empty(n, np.int64); tx = np.empty(n, np.int64)
    td = np.empty(n, np.int64); tp = np.empty(n); tc = np.empty(n)
    for i in range(n):
        want = 0
        if Ls[i] and pos <= 0 and (pos == 0 or i - ei >= hold): want = 1
        elif Ss[i] and pos >= 0 and (pos == 0 or i - ei >= hold): want = -1
        if want != 0:
            px = c[i] + tick if want == 1 else c[i] - tick
            if pos != 0:
                fee = (ep + px) * q * comm; pnl = (px - ep) * q * pos - fee
                real += pnl; te[k] = ei; tx[k] = i; td[k] = pos; tp[k] = pnl; tc[k] = fee; k += 1
            q = np.floor(cash / px); ep = px; pos = want; ei = i
        eq[i] = real + ((c[i] - ep) * q * pos - ep * q * comm if pos != 0 else 0.0)
    if pos != 0:                                                # mark open trade at last close
        px = c[n-1]; fee = (ep + px) * q * comm
        te[k] = ei; tx[k] = n - 1; td[k] = pos; tp[k] = (px - ep) * q * pos - fee; tc[k] = fee; k += 1
    return eq, te[:k], tx[:k], td[:k], tp[:k], tc[:k]

# ---------- metrics on MTM equity ----------
DAY = pd.Series(np.arange(N), index=B5.index).groupby(B5.index.floor('D')).last()
DAY_I, DAY_T = DAY.values, DAY.index
MON = pd.Series(np.arange(N), index=B5.index).groupby(B5.index.tz_localize(None).to_period('M')).last().values
I_SPLIT = int(np.searchsorted(B5.index, SPLIT))

def metrics(eq, tx, td, tp, tc, i0=0, i1=N, n_min=30):
    base = eq[i0-1] if i0 > 0 else 0.0; e = eq[i0:i1] - base
    sel = (tx >= i0) & (tx < i1); p = tp[sel]
    di = DAY_I[(DAY_I >= i0) & (DAY_I < i1)]; ed = eq[di] - base
    yrs = (i1 - i0) / (288 * 365.25); net = e[-1]
    dd_bar = np.maximum.accumulate(np.maximum(e, 0)) - e; mdd = dd_bar.max()
    ddd = np.maximum.accumulate(np.maximum(ed, 0)) - ed
    p95, p98, p99 = np.percentile(ddd, [95, 98, 99])
    gp, gl = p[p > 0].sum(), -p[p <= 0].sum()
    r = np.diff(np.concatenate([[0.0], ed])) / CAP
    sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan
    dn = np.sqrt((np.minimum(r, 0) ** 2).mean()); so = r.mean() / dn * np.sqrt(365) if dn > 0 else np.nan
    mi = MON[(MON >= i0) & (MON < i1)]; mp = np.diff(np.concatenate([[0.0], eq[mi] - base]))
    t = np.arange(len(ed)); r2 = np.corrcoef(t, ed)[0, 1] ** 2 if len(ed) > 2 and ed.std() > 0 else 0
    cv = mp.std() / mp.mean() if mp.mean() > 0 else np.inf
    m = dict(net=net, pf=gp / gl if gl > 0 else np.inf, n=int(sel.sum()), win=100 * (p > 0).mean() if len(p) else 0,
             long=tp[sel & (td > 0)].sum(), short=tp[sel & (td < 0)].sum(), comm=tc[sel].sum(),
             sharpe=sh, sortino=so, calmar=net / yrs / mdd if mdd > 0 else np.inf,
             p95=p95, p98=p98, p99=p99, mdd=mdd, r2=r2, posm=(mp > 0).mean(), cv=cv)
    ok = net > 0 and m['pf'] >= 1.15 and m['n'] >= n_min
    m['score'] = (net / yrs / max(p98, 1e-9)) * r2 ** 2 * m['posm'] / (1 + 0.25 * min(cv, 8)) if ok else -np.inf
    return m

def evaluate(Ls, Ss, hold=0):
    res = run(C, Ls, Ss, hold, TICK, COMM, CASH)
    return res, metrics(*res[:1], *res[2:]), metrics(*res[:1], *res[2:], 0, I_SPLIT), metrics(*res[:1], *res[2:], I_SPLIT, N)
