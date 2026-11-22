import time, pickle, numpy as np, pandas as pd
from p1m import build, run, report
t0 = time.time(); D5 = build(5); D1 = build(1); print('1 dk veri', len(D1['c']), f'{time.time()-t0:.0f}s', flush=True)
pickle.dump(D1, open('D1.pkl', 'wb'), protocol=4)
rows = []
for w in ('tam', 'tv'):
    rows.append(report(D5, run(D5, whale=w), f'5 dk V6.6 · balina {w}'))
    t0 = time.time(); out = run(D1, whale=w); rows.append(report(D1, out, f'1 dk V6.6 (süre eşdeğeri) · balina {w}')); print(f'1 dk koşu {time.time()-t0:.0f}s', flush=True)
    if w == 'tam':
        ch = np.flatnonzero(np.diff(np.r_[0, out]) != 0); hold = np.diff(np.r_[ch, len(out)]); print('1 dk pozisyon değişimi', len(ch), '| ≤5 dk süren pozisyon %', round((hold <= 5).mean() * 100, 1))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
