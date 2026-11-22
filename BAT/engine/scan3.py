import numpy as np, pandas as pd, json, time, sys
from scan import *
from stopre import layer as slayer, ADX2, DI2, OI3, PX3
# GENEL TARAMA 2: konsorsiyum (ağırlık/eşik/bileşen) + poz içi zarar sınırı + geri giriş BİRLİKTE
CFG0 = json.load(open('v51_cfg.json')); CFG0['p'] = {int(k): v for k, v in CFG0['p'].items()}
def full(cfg, S, rA):
    tg = system(cfg); tg2, _ = slayer(D['c'], tg, ADX2, DI2, OI3, PX3, S, 1, rA, 0.0, 0.0, 0, TICK); return metrics(D, simR(D, tg2))
if __name__ == '__main__':
    VAR = list(variants()); groups = {}
    for lab, ch in VAR:
        key = (ch[0], ch[1] if ch[0] != 'p' else tuple(ch[1])); groups.setdefault(str(key), []).append(ch)
    G = list(groups.values()); rng = np.random.default_rng(int(sys.argv[1])); rows = []; t0 = time.time()
    while time.time() - t0 < float(sys.argv[2]):
        cfg = json.loads(json.dumps(CFG0)); cfg['p'] = {int(k): v for k, v in cfg['p'].items()}; chs = []
        for g in G:
            if rng.random() < 0.15:
                ch = g[rng.integers(len(g))]; cfg = apply(cfg, ch); chs.append(ch)
        S = [6.0, 7.0, 8.0, 10.0][rng.integers(4)]; rA = [2.0, 3.0, 4.0][rng.integers(3)]
        m = full(cfg, S, rA); rows.append(dict(chs=json.dumps(chs), S=S, rA=rA, **{k: m[k] for k in K}))
    df = pd.DataFrame(rows); df.to_pickle(f'scan3_{sys.argv[1]}.pkl'); print(len(df), 'varyant')
