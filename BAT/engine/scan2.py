import numpy as np, pandas as pd, json, time, sys
from scan import *
# GENİŞ GENEL TARAMA: tabandan rastgele çoklu komşu değişiklik (her parametre %20 olasılıkla komşu değere)
if __name__ == '__main__':
    b = ev(BASE); VAR = list(variants()); groups = {}
    for lab, ch in VAR:
        key = (ch[0], ch[1] if ch[0] != 'p' else tuple(ch[1])); groups.setdefault(str(key), []).append(ch)
    G = list(groups.values()); rng = np.random.default_rng(int(sys.argv[1])); rows = []; t0 = time.time()
    while time.time() - t0 < float(sys.argv[2]):
        cfg = json.loads(json.dumps(BASE)); cfg['p'] = {int(k): v for k, v in cfg['p'].items()}; chs = []
        for g in G:
            if rng.random() < 0.2:
                ch = g[rng.integers(len(g))]; cfg = apply(cfg, ch); chs.append(ch)
        if not chs: continue
        m = ev(cfg); rows.append(dict(chs=json.dumps(chs), **{k: m[k] for k in K}))
    df = pd.DataFrame(rows); df.to_pickle(f'scan2_{sys.argv[1]}.pkl'); print(len(df), 'çoklu varyant', f'{time.time()-t0:.0f}s')
    ok = df[(df.net >= b['net'] * 1.06) & (df.IS >= b['IS']) & (df.OOS >= b['OOS']) & (df.pf >= b['pf']) & (df.p98 <= b['p98'] * 1.1)]
    print('kabul (net ≥ +%6, iki dönem ↑, PF ↑, P98 ≤ %110):', len(ok))
    if len(ok):
        ok = ok.assign(denge=np.minimum(ok.IS / b['IS'], ok.OOS / b['OOS'])).sort_values('denge', ascending=False)
        pd.set_option('display.width', 250); print(ok.head(8)[['net', 'n', 'pf', 'sharpe', 'p98', 'ulcer', 'upi', 'IS', 'OOS', 'last12', 'denge']].to_string(index=False))
