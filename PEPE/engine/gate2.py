exec(open("gate.py").read().split("rows=[dict(")[0])
D=pd.read_pickle('gate.pkl');pd.set_option('display.width',300)
b=D.iloc[0];print('V10\'u kalite (net/MaxDD) ile geçen varyantlar:');print(D[(D.q>b.q)].round(2).to_string(index=False))
@numba.njit
def run_soft(c,sp,dd,thr,blk,cross,srcarr,cev,tick,comm,cash):
    n=c.size;pos=0;ep=0.;q=0.;ei=0;real=0.;eq=np.empty(n);kk=0;bu=-1;nb=0
    te=np.empty(n,np.int64);tx=np.empty(n,np.int64);td=np.empty(n,np.int64);tp=np.empty(n);tc=np.empty(n)
    for i in range(n):
        if thr[i]>0 and dd[i]>thr[i] and i>bu:
            if cross==0 or (i>0 and dd[i-1]<=thr[i]): bu=i+blk;nb+=1
        want=sp[i]
        if i<=bu:
            ok=(srcarr[i]==0 and sp[i]!=0) or (cev[i] and sp[i]!=0)
            if not ok: want=pos          # sıkıntıda: mevcut pozisyonu tut, varyant hamlesi yok
        if want!=pos:
            if pos!=0:
                px=c[i]-tick if pos>0 else c[i]+tick;fee=(ep+px)*q*comm;pnl=(px-ep)*q*pos-fee;real+=pnl;te[kk]=ei;tx[kk]=i;td[kk]=pos;tp[kk]=pnl;tc[kk]=fee;kk+=1
            if want!=0:
                px=c[i]+tick if want>0 else c[i]-tick;q=np.floor(cash/px);ep=px;ei=i
            pos=want
        eq[i]=real+((c[i]-ep)*q*pos-ep*q*comm if pos!=0 else 0.)
    if pos!=0:
        px=c[n-1];fee=(ep+px)*q*comm;te[kk]=ei;tx[kk]=n-1;td[kk]=pos;tp[kk]=(px-ep)*q*pos-fee;tc[kk]=fee;kk+=1
    return eq,te[:kk],tx[:kk],td[:kk],tp[:kk],tc[:kk],nb
rows=[dict(L=0,pq=0,blk=0,cross=0,blocks=0,**H2(V10))]
for L,qq,blk,cross in itertools.product([30,60,90,180],[95,98,99],[42,84,168],[1,0]):
    rp=pd.Series(ddd).rolling(L,min_periods=L).quantile(qq/100).shift(1).values;thr=np.nan_to_num(rp,nan=0.)[dayof]
    r=run_soft(C,sp,ddb,thr,blk,cross,src,coreEvt,TICK,COMM,CASH);rows.append(dict(L=L,pq=qq,blk=blk,cross=cross,blocks=r[6],**H2(r)))
E=pd.DataFrame(rows);pd.set_option('display.width',300)
print('YUMUŞAK KAPI (pozisyonu tut, varyant hamlesi yok) — 7 saat, P98:');print(E[(E.blk==84)&(E.pq==98)].round(2).to_string(index=False))
print('q>V10:',(E.q>E.q.iloc[0]).sum(),'/',len(E)-1,'| net>V10:',(E.net>E.net.iloc[0]).sum(),'| q medyan',round(E.q.iloc[1:].median(),2),'(V10',round(E.q.iloc[0],2),')')
E.to_pickle('gate2.pkl')
