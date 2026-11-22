import pandas as pd
from hold2 import ev, st5, A30
rows = [ev(st5(1.5, 10, 3.5) * (A30 >= 25), st5(1.5, 10, 3.5) * (A30 >= 25), lab='V6.5')]
for f, aH, aT, xl, lab in [(2.5, 25, 30, 999, 'B'), (2.25, 25, 30, 999, 'B çarpan 2.25'), (2.75, 25, 30, 999, 'B çarpan 2.75'), (2.5, 22, 30, 999, 'B tut ADX 22'), (2.5, 28, 30, 999, 'B tut ADX 28'),
                           (2.5, 25, 28, 999, 'B ters ADX 28'), (2.5, 25, 32, 999, 'B ters ADX 32'), (2.5, 25, 30, 2.0, 'B + tut yalnız zarar ≤ %2')]:
    s = st5(1.5, 10, f); rows.append(ev(s * (A30 >= aH), s * (A30 >= aT), xl, lab=lab))
s = st5(1.5, 10, 2.5, lag=2); rows.append(ev(s * (A30 >= 25), s * (A30 >= 30), lab='B +1 mum gecikme'))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
