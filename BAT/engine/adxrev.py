import numpy as np, numba as nb, pandas as pd, json, time, sys, itertools
from scan import *
# V5.1 + ANLIK YÖNSEL DÖNÜŞ: (zarar ≥ L1 ve ADX5 > a1 ve ADX30 > b1) VEYA (zarar ≥ L2 ve ADX5 > a2 ve ADX30 > b2) → ters yön
@nb.njit(cache=True)
def adxrev(c, tg, adx5, adx30, L1, a1, b1, L2, a2, b2, cSL, cBars, reapply, tick):
    n = len(c); out = np.zeros(n, np.int64); sh = 0; ep = 0.0; st = 0; cDir = 0; cPx = 0.0; cBar = 0; prev = 0; nflip = 0
    for i in range(n):
        t_ = tg[i]; evt = t_ != prev; prev = t_; w = sh
        if evt:
            st = 0; w = t_
        elif st == 0:
            w = t_
        else:
            g = (c[i] / cPx - 1) * (-cDir) * 100
            if g <= -cSL or i - cBar >= cBars:
                st = 0; w = t_
            else:
                w = -cDir
        if sh != 0 and w == sh and (st == 0 or reapply == 1):
            loss = (c[i] / ep - 1) * sh * 100; x5 = adx5[i]; x30 = adx30[i]
            if (not np.isnan(x5)) and (not np.isnan(x30)):
                if (loss <= -L1 and x5 > a1 and x30 > b1) or (loss <= -L2 and x5 > a2 and x30 > b2):
                    if st == 0:
                        w = -sh; st = 1; cDir = sh; cPx = c[i]; cBar = i; nflip += 1
                    else:
                        st = 0; w = t_ if t_ != 0 else -sh; nflip += 1
        if w != sh:
            sh = w; ep = c[i] + tick * w
        out[i] = w
    return out, nflip
CFG = json.load(open('v51_cfg.json')); CFG['p'] = {int(k): v for k, v in CFG['p'].items()}
def run_rule(tg, r):
    return adxrev(D['c'], tg, D['adx5'], D['adx30'], r['L1'], r['a1'], r['b1'], r['L2'], r['a2'], r['b2'], r['cSL'], r['cBars'], r['re'], TICK)
if __name__ == '__main__':
    tg0 = system(CFG); b = metrics(D, simR(D, tg0)); K2 = ['net', 'n', 'pf', 'sharpe', 'p98', 'mddc', 'ulcer', 'upi', 'IS', 'OOS', 'last12']
    print('BAZ V5.1', {k: b[k] for k in K2}, flush=True)
    U = dict(L1=4.0, a1=23.0, b1=22.0, L2=6.0, a2=17.0, b2=15.0); rows = []
    for cSL, cB, re in itertools.product([2.0, 3.0, 5.0, 99.0], [144, 288, 576, 10**9], [0, 1]):
        r = dict(U, cSL=cSL, cBars=cB, re=re); tg, nf = run_rule(tg0, r); m = metrics(D, simR(D, tg)); rows.append(dict(cSL=cSL, cBars=cB if cB < 10**9 else 'olay', tekrar=re, dönüş=nf, **{k: m[k] for k in K2}))
    df = pd.DataFrame(rows); pd.set_option('display.width', 250); print('SENİN KURALIN (4%/23/22 + 6%/17/15), çıkış varyantları:'); print(df.to_string(index=False), flush=True)
    df.to_pickle('adxrev_user.pkl')
