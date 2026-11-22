import pickle, numpy as np, pandas as pd
from p1m import run, report
D1 = pickle.load(open('D1.pkl', 'rb')); rows = []
for conf in (1, 2, 3, 5, 10): rows.append(report(D1, run(D1, conf=conf), f'yön değişimi {conf} dk teyit'))
for clk in (5, 15, 30): rows.append(report(D1, run(D1, clk=clk), f'karar saati {clk} dk'))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
