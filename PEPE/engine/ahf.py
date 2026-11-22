exec(open("v9.py").read().split("karr=np.zeros(N)")[0])
@numba.njit
def run_ahf(c,t,R,thr,tpc,slc,mc,rb,trim,X,Y,M,z,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;prev=0;mode=0;cdir=0;cxp=0.;cxb=0;xp=0.;xb=0;md=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        ce=(t[i]!=prev);prev=t[i];want=pos;tag=0
        if ce:
            mode=0;want=t[i]
        elif mode==0:
            want=t[i]
            if pos!=0 and pos==t[i]:
                adv=(c[i]/ep-1.0)*(-pos)
                if thr[i]>0 and adv>=thr[i]:
                    want=-pos;mode=1;cdir=pos;cxp=c[i];cxb=i;tag=4
                elif trim==1 and ((pos>0 and R[i]>=X) or (pos<0 and R[i]<=100-X)):
                    want=0;mode=2;xp=c[i];xb=i;md=pos;tag=1
        elif mode==1:
            g=(c[i]/cxp-1.0)*(-cdir)
            back=(tpc>0 and g>=tpc) or (g<=-slc) or (i-cxb>=mc) or (rb==1 and ((cdir>0 and R[i]<=10.0) or (cdir<0 and R[i]>=90.0)))
            if back:
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
R7=RS[7]
def THR(kind,v): return np.full(N,v/100.) if kind=='%' else v*ATRP/100.
rows=[]
for kind,v in [('%',3),('%',4),('%',5),('%',6),('%',8),('atr',2),('atr',3),('atr',4)]:
    thr=THR(kind,v)
    for tpc,slc,mc,rb,tr in itertools.product([.02,.04,.08,0.],[.015,.03,.05],[48,144,288],[0,1],[0,1]):
        r=run_ahf(C,t8,R7,thr,tpc,slc,mc,rb,tr,90.,60.,12,.03,TICK,COMM,CASH)
        f=metrics(r[0],*r[2:6]);i=metrics(r[0],*r[2:6],0,I_SPLIT);o=metrics(r[0],*r[2:6],I_SPLIT,N)
        rows.append(dict(kind=kind,v=v,tpc=tpc,slc=slc,mc=mc,rb=rb,tr=tr,net=f['net'],pct=f['net']/5,pf=f['pf'],n=f['n'],p99=f['p99'],mdd=f['mdd'],q=f['net']/f['mdd'],Inet=i['net'],Ipf=i['pf'],Imdd=i['mdd'],Onet=o['net'],Opf=o['pf'],Ocal=o['calmar'],posm=f['posm'],flips=int((r[6]==4).sum())))
D=pd.DataFrame(rows);D.to_pickle('ahf.pkl');pd.set_option('display.width',260)
D['Iq']=D.Inet/D.Imdd;D['Isel']=D.Inet/1000*np.sqrt(D.Iq)
print('referans V8 (tr0) / V9 (tr1):');print(D[(D.kind=='%')&(D.v==3)].head(0).to_string())
print('net>8104 sayısı',(D.net>8104).sum(),'| max net',round(D.net.max()),'| PF>1.6&net>8000',((D.pf>1.6)&(D.net>8000)).sum())
print(D.sort_values('Isel',ascending=False).head(15).round(2).to_string(index=False))
