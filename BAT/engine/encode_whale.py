import numpy as np, pandas as pd, json
from short_top_lib import DIVh, zh
A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
z30 = pd.Series(zh(DIVh, 720), index=DIVh.index); z24 = pd.Series(zh(DIVh, 24), index=DIVh.index)
L = ((z30 >= 1) | (z24 >= 1.5)).astype(int); S = ((z30 <= -1) | (z24 <= -1.5)).astype(int)
st = (L + 2 * S).shift(1).fillna(0).astype(int)                    # saat H'nin kapısı = H−1 saatinin değeri
st = st[st.index >= st[st > 0].index[0]]
H0 = int((st.index[0] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')); TOT = len(st)
runs = []; cur = st.iloc[0]; ln = 0
for v in st.values:
    if v == cur: ln += 1
    else: runs.append((cur, ln)); cur = v; ln = 1
runs.append((cur, ln))
def enc(s, n):
    out = ""
    while n > 0:
        m = min(n, 4095)
        out += A[s * 16 + m] if m <= 15 else A[s * 16] + A[m // 64] + A[m % 64]
        n -= m
    return out
chunks = []; cur = ""
for s, n in runs:
    e = enc(int(s), int(n))
    if len(cur) + len(e) > 900: chunks.append(cur); cur = ""
    cur += e
chunks.append(cur)
# Pine çözücüsünün Python ikizi → doğrulama
dec = []
for ch in chunks:
    i = 0
    while i < len(ch):
        v = A.index(ch[i]); s = v // 16; n = v % 16
        if n == 0: n = A.index(ch[i + 1]) * 64 + A.index(ch[i + 2]); i += 3
        else: i += 1
        dec += [s] * n
print('saat', TOT, '| koşu', len(runs), '| karakter', sum(map(len, chunks)), '| parça', len(chunks), '| çözüm birebir:', dec == list(st.values), '| H0', H0, st.index[0], '→', st.index[-1])
json.dump(dict(H0=H0, TOT=TOT, chunks=chunks), open('whale_code.json', 'w'))
