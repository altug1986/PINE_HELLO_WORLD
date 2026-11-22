import pandas as pd
from stlib import ev, st5
rows = [ev(None, lab='baz (TV V6.3)'), ev(None, full=True, lab='baz tam balina')]
for lab, h, f, kw in [('tut', 3, 3.0, dict(offW=-0.15)), ('ters eşik .12', 3, 2.0, dict(onCS=0.12))]:
    for tag, s, full in [('', st5(h, 10, f), False), (' · +1 mum gecikme', st5(h, 10, f, lag=2), False), (' · ATR14', st5(h, 14, f), False),
                         (' · çarpan −0.5', st5(h, 10, f - 0.5), False), (' · çarpan +0.5', st5(h, 10, f + 0.5), False), (' · tam balina', st5(h, 10, f), True)]:
        rows.append(ev(s, full=full, lab=f'{lab} {h}s/{f:g}{tag}', **kw))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
