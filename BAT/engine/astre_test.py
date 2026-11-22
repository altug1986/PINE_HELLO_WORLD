# Astre V6.2 balina bloğunun birebir Python portu → motor kapısıyla kıyas + V6.3 (REV-20 zaman-stop) koşusu
import numpy as np, pandas as pd, v55
from collections import deque
from short_top_lib import to5, zh, DIVh, q0, D, simR, run, evalp, m30, t
from teco import daily, mask_from
from eng import equity
from v55 import n
HMS = 3600000
d = pd.read_pickle('top5.pkl')
P = dict(zip(d.ms[d.sum_toptrader_long_short_ratio > 0], d.sum_toptrader_long_short_ratio[d.sum_toptrader_long_short_ratio > 0]))
A = dict(zip(d.ms[d.count_toptrader_long_short_ratio > 0], d.count_toptrader_long_short_ratio[d.count_toptrader_long_short_ratio > 0]))
H0 = (t[0] // HMS) * HMS; hours = np.arange(H0, (t[-1] // HMS) * HMS + HMS, HMS)
def astre(sh):                       # sh: H saatinde okunan anlık görüntünün create_time farkı
    V, T = deque(), deque(); last = None; st = np.zeros(len(hours), np.int8); miss = 0
    for k, H in enumerate(hours):
        p, a = P.get(H + sh), A.get(H + sh); ok = p is not None and a is not None; z = z24 = None; rep = False
        while T and T[0] <= H - 720 * HMS: T.popleft(); V.popleft()
        if ok:
            dv = np.log(p) - np.log(a); rep = last is not None and dv == last
            if not rep:
                V.append(dv); T.append(H)
                if len(V) >= 360:
                    x = np.fromiter(V, float); sd = x.std(ddof=1); z = (dv - x.mean()) / sd if sd > 0 else None
            last = dv
        cnt = 0
        for tt in reversed(T):
            if tt <= H - 24 * HMS: break
            cnt += 1
        if ok and not rep and cnt >= 12:
            x = np.array(list(V)[-cnt:]); sd = x.std(ddof=1); z24 = (V[-1] - x.mean()) / sd if sd > 0 else None
        gl = (z is not None and z >= 1) or (z24 is not None and z24 >= 1.5); gs = (z is not None and z <= -1) or (z24 is not None and z24 <= -1.5)
        st[k] = gl + 2 * gs
    j = (t // HMS * HMS - H0) // HMS; s = st[j]; return (s == 1) | (s == 3), (s == 2) | (s == 3)
a_ = np.nan_to_num(to5(zh(DIVh, 720))); b_ = np.nan_to_num(to5(zh(DIVh, 24))); WL = (a_ >= 1) | (b_ >= 1.5); WS = (a_ <= -1) | (b_ <= -1.5)
def R_(GL, GS, p): v55.GL = GL.astype(np.int64); v55.GS = GS.astype(np.int64); q = dict(q0, **p); return q, simR(D, run(q)[0])
_, R0 = R_(WL, WS, {}); M = mask_from(daily(R0), 7, 3)
def ev(GL, GS, p, lab):
    q, R = R_(GL, GS, p); e = evalp(q); m = m30(R); o = R['open']
    cl, mtm, worst = equity(n, R['eb'], R['xb'], R['dir'], R['ep'], R['q'], R['pnl'], D['h'], D['l'], D['c'], o[0], o[1], o[2], o[3])
    E = pd.Series(mtm, index=pd.to_datetime(t + 300000, unit='ms')).resample('30min').last().ffill(); dd = E.diff().fillna(0)
    inm = M.reindex(dd.index.normalize()).fillna(False).to_numpy(); yrs = (t[-1] - t[0]) / 3.15576e10; nd = dd.where(~inm, 0.0); eq = nd.cumsum(); uw = eq.cummax() - eq
    return {'sürüm': lab, 'net': e['net'], 'işlem': e['n'], 'IS': e['IS'], 'OOS': e['OOS'], 'son12ay': e['L12'], 'son200g': e['L200'], 'MaxDD30': m['MaxDD 30dk mum içi'],
            'Calmar30': m['Calmar 30dk'], 'DD P95/98/99': m['DD P95/P98/P99 30dk'], 'normal net': round(nd.sum()), 'normal Calmar': round(nd.sum() / yrs / uw.max(), 2)}
TS = dict(H=288, xH=-1.0, aH=25.0, rH=4.0)
if __name__ == '__main__':
    print('q0 rH', q0.get('rH'), 'rA', q0.get('rA'))
    rows = [ev(WL, WS, {}, 'V6.1 motor kapısı'), ev(WL, WS, TS, 'V6.2 motor kapısı (gömülü kod)')]
    for sh, nm in [(-300000, 'B: TV mumu açılış=create_time'), (0, 'A: TV mumu kapanış=create_time')]:
        GL, GS = astre(sh); same = ((GL == WL) & (GS == WS)).mean() * 100
        print(f'{nm}: kapı durumu motorla aynı bar %{same:.2f} | long engelli bar motor %{WL.mean()*100:.1f} astre %{GL.mean()*100:.1f} | short engelli motor %{WS.mean()*100:.1f} astre %{GS.mean()*100:.1f}')
        rows += [ev(GL, GS, {}, f'Astre V6.2 · {nm[:1]}'), ev(GL, GS, TS, f'Astre + zaman-stop (V6.3) · {nm[:1]}')]
    # :55 anlık görüntüsü eksik olduğu için Astre'nin kapıyı kapattığı saat sayısı
    hv = DIVh.to_numpy(); hk = ((DIVh.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')).to_numpy(np.int64)
    miss = sum(1 for H, v in zip(hk, hv) if not np.isnan(v) and (P.get(H + 3300000) is None or A.get(H + 3300000) is None))
    print('motorda değer var, Astre (B) için saat sonu :55 görüntüsü yok:', miss, 'saat /', int((~np.isnan(hv)).sum()))
    pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
