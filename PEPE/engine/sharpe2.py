exec(open("sharpe.py").read().split("rows=[dict(var='V10'")[0])
@numba.njit
def run_v11(c,t,R,thr,ps,sT,sC,slc,mc,X,Y,M,z,tick,comm,cash):
    # sT: trend Sharpe pozisyon yönünde >= sT ise mikro kâr alma yok (hareketle kal); sT<0 → filtre kapalı
    # sC: karşı poza geçiş için fiyat Sharpe'ı karşı yönde >= sC olmalı (pozitif teyit); sC<0 → filtre kapalı
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;prev=0;mode=0;cdir=0;cxp=0.;cxb=0;xp=0.;xb=0;md=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n);ts=np.zeros(n,np.int64)
    for i in range(n):
        ce=(t[i]!=prev);prev=t[i];want=pos;tag=0
        if ce:
            mode=0;want=t[i]
        elif mode==0:
            want=t[i]
            if pos!=0 and pos==t[i]:
                adv=(c[i]/ep-1.0)*(-pos);trs=ps[i]*pos
                if thr[i]>0 and adv>=thr[i] and (sC<0 or -trs>=sC):
                    want=-pos;mode=1;cdir=pos;cxp=c[i];cxb=i;tag=4
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
t10=chain(ENG);thr=4.5*ATRP/100.
PS={}
for W in [144,288,576,864,2016]:
    m=pd.Series(ret).rolling(W).mean();sd=pd.Series(ret).rolling(W).std();PS[W]=np.nan_to_num((m/sd*np.sqrt(W)).values,nan=0.)
R=lambda ps,sT,sC:run_v11(C,t10,R7,thr,ps,sT,sC,.03,144,90.,60.,12,.03,TICK,COMM,CASH)
rows=[dict(var='V10',W=0,sT=-1,sC=-1,**H(R(PS[288],-1.,-1.)))]
for W,sT in itertools.product(PS,[0.5,1.0,1.5,2.0,3.0]): rows.append(dict(var='C1 kâr al süzgeci',W=W,sT=sT,sC=-1,**H(R(PS[W],sT,-1.))))
for W,sC in itertools.product(PS,[0.,0.5,1.0,1.5]): rows.append(dict(var='C2 karşı poz teyidi',W=W,sT=-1,sC=sC,**H(R(PS[W],-1.,sC))))
D=pd.DataFrame(rows);D.to_pickle('sharpe2.pkl');pd.set_option('display.width',300)
b=D.iloc[0];print(D.iloc[:1].round(2).to_string(index=False))
g=D[(D.net>b.net)&(D.q>=b.q*0.97)&(D.Ocal>=b.Ocal*0.95)]
print('net↑ + kalite + OOS korunur:',len(g),'/',len(D)-1);print(g.sort_values('net',ascending=False).round(2).to_string(index=False))
for v in ['C1 kâr al süzgeci','C2 karşı poz teyidi']:
    s=D[D['var']==v];print(v,'net (satır W, sütun eşik):');print(s.pivot_table(index='W',columns='sT' if v.startswith('C1') else 'sC',values='net').round(0))
