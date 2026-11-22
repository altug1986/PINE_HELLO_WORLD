import numpy as np, pandas as pd
from p1m import build, run, report
rows = []
for tf in (5, 4, 3):
    D = build(tf)
    for w in ('tam', 'tv'): rows.append(dict(report(D, run(D, whale=w), f'{tf} dk V6.6 · balina {w}'), mum=len(D['c'])))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
