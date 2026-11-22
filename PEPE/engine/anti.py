exec(open('robust.py').read().split("\nCANDS=[]")[0])
import time;t0=time.time()
@numba.njit
def run_v14(c,t,R,thr,thr1,efl,ps,sT,ex,slc,mc,X,Y,M,z,tick,comm,cash,hmin,tl):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;prev=0;mode=0;cdir=0;cxp=0.;cxb=0;xp=0.;xb=0;md=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        ce=(t[i]!=prev);prev=t[i];want=pos;tag=0
        if ce and not (tl==1 and mode==2):
            mode=0;want=t[i]
        elif mode==0:
            want=t[i]
            if pos!=0 and pos==t[i]:
                adv=(c[i]/ep-1.0)*(-pos);trs=ps[i]*pos
                if (thr[i]>0 and adv>=thr[i]) or (thr1[i]>0 and adv>=thr1[i] and efl[i]==-pos):
                    want=-pos;mode=1;cdir=pos;cxp=c[i];cxb=i;tag=4
                elif ex[i]==pos:
                    want=0;mode=2;xp=c[i];xb=i;md=pos;tag=6
                elif ((pos>0 and R[i]>=X) or (pos<0 and R[i]<=100-X)) and (sT<0 or trs<sT):
                    want=0;mode=2;xp=c[i];xb=i;md=pos;tag=1
        elif mode==1:
            g=(c[i]/cxp-1.0)*(-cdir)
            if g<=-slc or i-cxb>=mc:
                mode=0;want=t[i];tag=5
            else:
                want=-cdir
        else:
            if md>0: back=(R[i]<=Y) or (i-xb>=M) or (c[i]>=xp*(1+z))
            else: back=(R[i]>=100-Y) or (i-xb>=M) or (c[i]<=xp*(1-z))
            if back:
                mode=0;want=t[i];tag=2
            else:
                want=0
        if hmin>0 and pos!=0 and want!=pos and i-ei<hmin and tag!=4:
            want=pos
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl
                te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;ts[kk]=tag;kk+=1
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],ts[:kk]
@numba.njit
def deb(x,k):
    out=np.zeros_like(x);cur=0;run=0;last=0
    for i in range(x.size):
        if x[i]==last: run+=1
        else: last=x[i];run=1
        if run>=k: cur=x[i]
        out[i]=cur
    return out
def chainX(E,pl):
    t=np.zeros(N,np.int64)
    for k in ORDER[::-1]: t=np.where(E[k]!=0,E[k],t)
    return np.where(pl!=0,pl,t).astype(np.int64)
def RUN4(t,ex,hmin=0,tl=0,stress=False,thr1=ZF,efl=ZI):
    return run_v14(C,t,R7,thr,thr1,efl,ps,2.0,ex,.03,144,90.,60.,12,.03,3*TICK if stress else TICK,.0008 if stress else COMM,CASH,hmin,tl)
def churn(r):
    te,tx,td,tp=r[1],r[2],r[3],r[4];d=tx-te;b=d<=2
    ff=sum(1 for j in range(len(te)-1) if d[j]<=2 and td[j+1]==-td[j] and te[j+1]-tx[j]<=1)
    return dict(blip=int(b.sum()),blip_net=float(tp[b].sum()),ff=ff)
def EV(t,ex,**k):
    r=RUN4(t,ex,**k);h=HH(r);h.update(churn(r));rs=RUN4(t,ex,stress=True,**k);hs=HH(rs);h.update(s_net=hs['net'],s_upi=hs['upi'],s_oos=hs['OOS_upi']);return h
# regresyon: v14(0,0) == v13
do0,ex0=comps(0);pl0=vstate(C,(dp6<=-4)&(do0<=-4),(dp6>=4)&(do0<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
r13=RUN(chain2(pl0),ex0);r14=RUN4(chain2(pl0),ex0);print('regresyon net',round(r13[4].sum(),2),round(r14[4].sum(),2))
# blip kaynakları (s=0 baz)
t0_=chain2(pl0);src=np.full(N,'yok',object)
for k in ORDER[::-1]: src=np.where(ENG[k]!=0,k,src)
src=np.where(pl0!=0,'PL',src);d=r14[2]-r14[1];bm=d<=2
print(pd.DataFrame({'src':src[r14[1][bm]],'pnl':r14[4][bm]}).groupby('src').pnl.agg(['count','sum']).round(1).to_string())
CANDS=[('A0 baz',{},None),('A1 MIK teyit2',{},'mik2'),('A1 MIK teyit3',{},'mik3'),('A2 RSIX+MIK teyit2',{},'rm2'),('A3 hedef teyit2',{},'all2')]
for h in [2,3,4,6]: CANDS.append((f'A4 min tutma {h}',dict(hmin=h),None))
CANDS+= [('A5 trim kilidi korunur',dict(tl=1),None),('A6 kilit+MIK teyit2',dict(tl=1),'mik2'),('A7 kilit+min3',dict(tl=1,hmin=3),None),('A7 kilit+min2',dict(tl=1,hmin=2),None),('A8 kilit+min3+MIK2',dict(tl=1,hmin=3),'mik2')]
rows=[]
for s in [0,1,2]:
    do,ex=comps(s);pl=vstate(C,(dp6<=-4)&(do<=-4),(dp6>=4)&(do<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
    for nm,kw,dv in CANDS:
        E=dict(ENG)
        if dv=='mik2': E['MIK']=deb(ENG['MIK'].astype(np.int64),2)
        if dv=='mik3': E['MIK']=deb(ENG['MIK'].astype(np.int64),3)
        if dv=='rm2':
            for kk in ['RSIX','MIK']: E[kk]=deb(ENG[kk].astype(np.int64),2)
        t=chainX(E,pl)
        if dv=='all2': t=deb(t,2)
        rows.append(dict(h=nm,s=s,**EV(t,ex,**kw)))
D=pd.DataFrame(rows);D.to_pickle('anti.pkl');print('süre',round(time.time()-t0),'s')
pd.set_option('display.width',300)
c=['net','pf','n','mdd','upi','IS_upi','OOS_upi','y26','blip','blip_net','ff','s_net','s_upi','s_oos']
for s in [0,1,2]: print(f'--- OI kayma {s} ---');print(D[D.s==s].set_index('h')[c].round(2).to_string())
