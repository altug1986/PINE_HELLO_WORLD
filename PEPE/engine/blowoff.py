exec(open('ptrail.py').read().split("def health")[0].split("x=pd.ExcelFile")[0]+open('ptrail.py').read().split("print('TV işlem'")[1].split("\n",1)[1].split("@numba.njit")[0])
exec("def health"+open('ptrail.py').read().split("def health")[1].split("rows=[")[0])
h1,l1,c1,th1,_=htf(60);pc=np.r_[np.nan,c1[:-1]];tr1=np.fmax(np.fmax(h1-l1,np.abs(h1-pc)),np.abs(l1-pc));atr=np.nan_to_num(htf_map(rma(tr1,14)/c1,th1,3600),nan=.02)
@numba.njit
def ov(c,atr,eq,te,tx,td,K,G,frac,tick,cash):
    n=c.size;dl=np.zeros(n);tpn=np.zeros(te.size);cnt=0
    for k in range(te.size):
        e=te[k];x=tx[k];d=td[k];ep=c[e]+tick*d;q=np.floor(cash/ep)*frac;pk=0.;hit=-1;px=0.;a=atr[e]
        for i in range(e+1,x):
            r=(c[i]/ep-1)*d
            if r>pk: pk=r
            if pk>=K*a and pk-r>=G*a:
                hit=i;px=c[i]-tick*d;break
        if hit>=0:
            cnt+=1;xp=c[x]-tick*d
            for i in range(hit+1,x): dl[i]-=q*d*(c[i]-px)
            dv=q*d*(px-xp)
            for i in range(x,n): dl[i]+=dv
            tpn[k]=dv
    return eq+dl,tpn,cnt
rows=[dict(K=0,G=0,fr=0,hits=0,**health(eq,tp))]
for K,G,fr in itertools.product([4,6,8,10,12,15],[1,1.5,2,3,4],[.3,.5,1.]):
    if G>=K: continue
    en,dv,cnt=ov(C,atr,eq,te,tx,td,K,G,fr,TICK,CASH);rows.append(dict(K=K,G=G,fr=fr,hits=cnt,**health(en,tp+dv)))
D=pd.DataFrame(rows);D.to_pickle('blowoff.pkl');pd.set_option('display.width',250);b=D.iloc[0]
for fr in [.3,.5,1.]:
    s=D[D.fr==fr];print(f'kilit oranı %{fr*100:.0f} — net farkı (satır K×ATR kâr, sütun G×ATR geri çekilme):');print((s.pivot_table(index='K',columns='G',values='net')-b.net).round(0))
g=D[(D.net>b.net)&(D.q>=b.q)&(D.Ocal>=b.Ocal*0.97)];print('net↑ kalite↑ OOS korunur:',len(g),'/',len(D)-1)
print(g.sort_values('net',ascending=False).head(8).round(2).to_string(index=False))
for K,G in [(8,2),(12,2),(6,3)]:
    _,dv,cnt=ov(C,atr,eq,te,tx,td,K,G,1.,TICK,CASH);h=dv[dv!=0]
    print(f'K{K} G{G}: tetik {cnt} | tepeyi yakaladı (kazandırdı) {(h>0).sum()} işlem +{h[h>0].sum():.0f}$ | trend devam etti (kaybettirdi) {(h<0).sum()} işlem {h[h<0].sum():.0f}$ | en büyük kaçırılan {h.min():.0f}$')
