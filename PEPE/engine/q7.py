import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
exec("@numba.njit\n"+open('arge2.py').read().split("@__import__('numba').njit\n")[1].split("def rec_t")[0])
exec(open('arge5.py').read().split("@numba.njit\ndef vstate")[1].split("F=np.zeros")[0].join(["@numba.njit\ndef vstate",""]))
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
def pos(r):
    p=np.zeros(N,np.int64)
    for e,x,d in zip(r[1],r[2],r[3]): p[e:x]=d
    if len(r[1]): p[r[1][-1]:]=r[3][-1]
    return p
S=pickle.load(open('S.pkl','rb'));PR=pickle.load(open('/home/claude/data/PR.pkl','rb'));V6=pickle.load(open('V6.pkl','rb'))
t4=np.sign(pos(S['A=V2'])+pos(S['V3']));pR=PR[(14,10,50,0.08,144)];t6=np.where(pR!=0,pR,t4)
F=np.zeros(N,bool)
# monthly V6 + regime look
mo=pd.Series(V6[0][DAY_I],index=DAY_T).resample('ME').last().diff()
print('V6 aylık net medyan',round(mo.median(),1),'| yıl medyanları',mo.groupby(mo.index.year).median().round(1).to_dict())
print('son 8 ay',mo.tail(8).round(0).to_dict())
# HTF regime features (last completed HTF bar)
A60,_=adx_htf(60);A240,_=adx_htf(240)
def chop_htf(tf,n=14):
    h,l,c,th,s=htf(tf);tr=np.maximum(np.maximum(h-l,np.abs(h-np.r_[np.nan,c[:-1]])),np.abs(l-np.r_[np.nan,c[:-1]]))
    st=pd.Series(tr).rolling(n).sum().values;rng=pd.Series(h).rolling(n).max().values-pd.Series(l).rolling(n).min().values
    return htf_map(100*np.log10(st/rng)/np.log10(n),th,s)
CH60,CH240=chop_htf(60),chop_htf(240)
for nm,x in [('ADX1h',A60),('ADX4h',A240),('CHOP1h',CH60),('CHOP4h',CH240)]: print(nm,'P20/50/80',np.nanpercentile(x,[20,50,80]).round(1))
d=np.diff(C,prepend=np.nan)
def rsi(n): u=rma(np.where(d>0,d,0.),n);w=rma(np.where(d<0,-d,0.),n);return 100-100/(1+u/w)
R14=rsi(14);R7=rsi(7)
bb_m=pd.Series(C).rolling(20).mean().values;bb_s=pd.Series(C).rolling(20).std(ddof=0).values
K=['net','pf','n','calmar','p98','mdd','sharpe','sortino','r2','posm']
rows=[]
def rec(fam,prm,t):
    r=run_t(C,t.astype(np.int64),TICK,COMM,CASH);f=metrics(r[0],*r[2:]);i=metrics(r[0],*r[2:],0,I_SPLIT);o=metrics(r[0],*r[2:],I_SPLIT,N)
    m=pd.Series(r[0][DAY_I],index=DAY_T).resample('ME').last().diff();ym=m.groupby(m.index.year).median()
    l6=metrics(r[0],*r[2:],np.searchsorted(B5.index,pd.Timestamp('2026-05-19',tz='UTC')),N)
    rows.append(dict(fam=fam,**prm,**{k:f[k] for k in K},nDD=f['net']/max(f['mdd'],1),Ical=i['calmar'],Ocal=o['calmar'],Onet=o['net'],medM=m.median(),minYmed=ym.min(),l6=l6['net']))
rec('V6',{},t6)
REG={}
for nm,x,th in [('ADX1h',A60,[15,20,25]),('ADX4h',A240,[15,20,25]),('CHOP1h',CH60,[50,55,61.8]),('CHOP4h',CH240,[50,55,61.8])]:
    for v in th: REG[(nm,v)]=(x<v) if 'ADX' in nm else (x>v)
ENG={}
for lo,sl,mb in itertools.product([20,25,30],[.03,.05],[48,144]):
    ENG[('RSI14',lo,sl,mb)]=(R14<lo,R14>100-lo,R14>50,R14<50,sl,mb)
for lo,sl,mb in itertools.product([15,20],[.03,.05],[48]): ENG[('RSI7',lo,sl,mb)]=(R7<lo,R7>100-lo,R7>50,R7<50,sl,mb)
for k,sl in itertools.product([2.0,2.5,3.0],[.03,.05]): ENG[('BB',k,sl,48)]=(C<bb_m-k*bb_s,C>bb_m+k*bb_s,C>=bb_m,C<=bb_m,sl,48)
for (rn,rv),g in REG.items():
    for ek,(L,S_,xl,xs,sl,mb) in ENG.items():
        pm=vstate(C,g&L,g&S_,0.,sl,mb,xl,xs,0)
        rec('FILL',dict(reg=f'{rn}{rv}',eng=str(ek)),np.where(pR!=0,pR,np.where(t4!=0,t4,pm)))          # V4 flat iken
        rec('OVR',dict(reg=f'{rn}{rv}',eng=str(ek)),np.where(pR!=0,pR,np.where(pm!=0,pm,t4)))           # yatayda MR öncelikli
D=pd.DataFrame(rows);D.to_pickle('q7.pkl');pd.set_option('display.width',260)
print(D.iloc[:1].round(2).to_string(index=False))
for fam in ['FILL','OVR']:
    g=D[(D.fam==fam)&(D.Ical>D.Ical.iloc[0]*0.95)&(D.Onet>D.Onet.iloc[0])];print(fam,len(g),'/',(D.fam==fam).sum(),'IS kalite korunur ve OOS net↑')
    print(g.sort_values('nDD',ascending=False).head(8).round(2).to_string(index=False))
