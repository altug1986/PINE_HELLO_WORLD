import pickle, itertools, numpy as np, pandas as pd, numba
from eng import *
exec(open('arge5.py').read().split("F=np.zeros")[0].split("from eng import *")[1])
F=np.zeros(N,bool)
@numba.njit
def run_pyr(c,t,aL,aS,K,gap,needp,tick,comm,cash):
    n=c.size;pos=0;nu=0;real=0.;eq=np.empty(n);ep=np.zeros(K);q=np.zeros(K);ei=np.zeros(K,np.int64);last=0
    te=np.empty(n*K,np.int64);tx=np.empty(n*K,np.int64);td=np.empty(n*K,np.int64);tp=np.empty(n*K);tc=np.empty(n*K);k=0;uh=np.zeros(K+1,np.int64)
    for i in range(n):
        if t[i]!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick
                for j in range(nu):
                    fee=(ep[j]+px)*q[j]*comm;pnl=(px-ep[j])*q[j]*pos-fee;real+=pnl;te[k]=ei[j];tx[k]=i;td[k]=pos;tp[k]=pnl;tc[k]=fee;k+=1
                nu=0
            if t[i]!=0:
                px=c[i]+tick if t[i]>0 else c[i]-tick;ep[0]=px;q[0]=np.floor(cash/px);ei[0]=i;nu=1;last=i
            pos=t[i]
        elif pos!=0 and nu<K and i-last>=gap and ((pos>0 and aL[i]) or (pos<0 and aS[i])):
            op=0.
            for j in range(nu): op+=(c[i]-ep[j])*q[j]*pos
            if needp==0 or op>0:
                px=c[i]+tick if pos>0 else c[i]-tick;ep[nu]=px;q[nu]=np.floor(cash/px);ei[nu]=i;nu+=1;last=i
        o=0.
        for j in range(nu): o+=(c[i]-ep[j])*q[j]*pos-ep[j]*q[j]*comm
        eq[i]=real+o;uh[nu]+=1
    if pos!=0:
        px=c[n-1]
        for j in range(nu):
            fee=(ep[j]+px)*q[j]*comm;te[k]=ei[j];tx[k]=n-1;td[k]=pos;tp[k]=(px-ep[j])*q[j]*pos-fee;tc[k]=fee;k+=1
    o=np.argsort(tx[:k],kind='mergesort')
    return eq,te[:k][o],tx[:k][o],td[:k][o],tp[:k][o],tc[:k][o],uh
gA=(a30>28)&(a5>20)&(a5>lag(a5,12));mA=mom(4);gB=(a30>25)&(a5>lag(a5,6));mB=mom(48)
SIG={'AB':((gA&(mA>=3))|(gB&(mB>=2.5)),(gA&(mA<=-3))|(gB&(mB<=-2.5))),'M12':(mom(12)>=1.5,mom(12)<=-1.5)}
hh,ll=donch(48);SIG['BRK48']=(C>hh,C<ll)
g=np.ones(N,bool);pc=vstate(C,mom(3)>=1.5,mom(3)<=-1.5,0.,0.,0,F,F,24);BASES={'V4':t4,'V4+MF':np.where(t4!=0,t4,pc)}
K_=['net','pf','n','sharpe','sortino','calmar','p95','p98','p99','mdd','r2','posm','cv']
rows=[]
for bn,bt in BASES.items():
    for sn,(L,S_) in SIG.items():
        for Kk,gap,npf in itertools.product([1,2,3,4,5],[12,24,48,96],[0,1]):
            if Kk==1 and (gap!=12 or npf or sn!='AB'): continue
            r=run_pyr(C,bt,L,S_,Kk,gap,npf,TICK,COMM,CASH);f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N)
            uh=r[6];avgU=(uh*np.arange(Kk+1)).sum()/N
            rows.append(dict(base=bn,sig=sn,K=Kk,gap=gap,np=npf,**{k:f[k] for k in K_},avgU=avgU,Onet=o['net'],Opf=o['pf'],Inet=i['net'],netDD=f['net']/f['mdd']))
D=pd.DataFrame(rows);D.to_pickle('pyr.pkl');pd.set_option('display.width',260)
print(D[D.K==1].round(2).to_string(index=False))
for Kk in [3,4,5]:
    s=D[(D.K==Kk)&(D.Inet>0)&(D.Onet>0)];print('K',Kk);print(s.sort_values('calmar',ascending=False).head(6).round(2).to_string(index=False))
