exec(open('vhf.py').read().split("\nCF=[]")[0])
CF=[]
for src,q,aq,oc,x in itertools.product(['yok','5m24','5m48','5m96','1h14'],[.1,.2,.3],[.7,.8,.9],['up','dn','abs'],[1,2,3]):
    if src=='yok' and q!=.1: continue
    CF.append(('G1 yönsüz+vol+OI30 flat',f'{src} q{q} ATR≥p{aq} OI {oc} {x}%',dict(k=1,src=src,q=q,aq=aq,oc=oc,x=x)))
for y,x,fl in itertools.product([.5,1,1.5,2,3],[.5,1,1.5,2,3],['yok','testere','yüksek vol']): CF.append(('G2 OI30 karşı birikim kilidi',f'fiyat ters≥{y}% OI≥{x}% filtre:{fl}',dict(k=2,y=y,x=x,fl=fl)))
for k1,x,y in itertools.product([2,2.5,3,3.5],[.5,1,2,3],[0,.5,1]): CF.append(('G3 OI30 teyitli erken karşı',f'k{k1} OI≥{x}% fiyat≥{y}%',dict(k=3,k1=k1,x=x,y=y)))
for z_,dr,aq in itertools.product([2,2.5,3,4],['abs','up','dn'],[None,.7,.8,.9]): CF.append(('G4 OI30 aşırı hareket flat',f'|z|≥{z_} {dr} ATR≥p{aq}',dict(k=4,z=z_,dr=dr,aq=aq)))
print('aday',len(CF))
rows=[];BASE={}
for s in [0,1,2]:
    do,ex=comps(s);pl=vstate(C,(dp6<=-4)&(do<=-4),(dp6>=4)&(do<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
    zo=np.nan_to_num(do/pd.Series(do).rolling(864,min_periods=200).std().values)
    tb=chainY(E0,pl);BASE[s]=HH(R6(tb,ex));rows.append(dict(f='BAZ V11.2',p='',s=s,**BASE[s]))
    for f,p,c in CF:
        k=c['k']
        if k==1:
            v=VH.get(c['src']);chop=np.ones(N,bool) if v is None else v<QI(v,c['q']);vol=ATRP>=QI(ATRP,c['aq'])
            oc=(do>=c['x']) if c['oc']=='up' else ((do<=-c['x']) if c['oc']=='dn' else (np.abs(do)>=c['x']))
            h=HH(R6(np.where(chop&vol&oc&(pl==0),0,tb).astype(np.int64),ex))
        elif k==2:
            fl=np.ones(N,bool) if c['fl']=='yok' else (VH['5m48']<QI(VH['5m48'],.3) if c['fl']=='testere' else ATRP>=QI(ATRP,.8))
            e2=np.where(ex!=0,ex,np.where(fl&(dp6<=-c['y'])&(do>=c['x']),1,np.where(fl&(dp6>=c['y'])&(do>=c['x']),-1,0))).astype(np.int64)
            h=HH(R6(tb,e2))
        elif k==3:
            efl=np.where((dp6<=-c['y'])&(do>=c['x']),-1,np.where((dp6>=c['y'])&(do>=c['x']),1,0)).astype(np.int64)
            h=HH(RUN5(tb,ex,thr1=c['k1']*ATRP/100.,efl=efl))
        else:
            zz=np.abs(zo)>=c['z'] if c['dr']=='abs' else ((zo>=c['z']) if c['dr']=='up' else (zo<=-c['z']))
            vol=np.ones(N,bool) if c['aq'] is None else ATRP>=QI(ATRP,c['aq'])
            h=HH(R6(np.where(zz&vol&(pl==0),0,tb).astype(np.int64),ex))
        rows.append(dict(f=f,p=p,s=s,**h))
D=pd.DataFrame(rows);D.to_pickle('oi30.pkl');print('süre',round(time.time()-t0),'s')
for s in [0,1,2]: b=BASE[s];print(f"baz s={s}: net {b['net']:.0f} PF {b['pf']:.3f} MaxDD {b['mdd']:.0f} UPI {b['upi']:.2f}")
X=D[D.f!='BAZ V11.2'].copy()
for m in ['net','upi','IS_upi','OOS_upi','mdd','pf']: X['d_'+m]=X.apply(lambda r:r[m]-BASE[r.s][m],axis=1)
X['ok']=(X.d_upi>0)&(X.d_net>=0)&(X.d_OOS_upi>=0)&(X.d_IS_upi>=0);X['okdd']=(X.d_mdd<0)&(X.d_net>=0)
x0=X[X.s==0];print(f"HEDEF (s=0) net≥9200 & PF≥1.65: {int(((x0.net>=9200)&(x0.pf>=1.65)).sum())} | net≥9200: {int((x0.net>=9200).sum())} | PF≥1.65: {int((x0.pf>=1.65).sum())} | max net {x0.net.max():.0f} | max PF {x0.pf.max():.3f}")
G=X.groupby(['f','p'],sort=False).agg(ok3=('ok','all'),dd3=('okdd','all'),net0=('net','first'),pf0=('pf','first'),mdd0=('mdd','first'),dnet_min=('d_net','min'),dmdd_max=('d_mdd','max'),dupi_min=('d_upi','min'),doos_min=('d_OOS_upi','min')).reset_index()
for f,g in G.groupby('f',sort=False): print(f'{f}: {len(g)} ayar | 3 kaymada UPI/net/IS/OOS↑ {int(g.ok3.sum())} | 3 kaymada MaxDD↓&net≥ {int(g.dd3.sum())} | Δnet(s0) medyan {(g.net0-BASE[0]["net"]).median():.0f}')
pd.set_option('display.width',300)
print('\n3 kaymada MaxDD↓ & net≥ (en çok MaxDD düşüren):');print(G[G.dd3].sort_values('dmdd_max').head(8).round(3).to_string(index=False))
print('\n3 kaymada UPI/net/IS/OOS↑:');print(G[G.ok3].sort_values('dupi_min',ascending=False).head(8).round(3).to_string(index=False))
print('\ns=0 en yüksek PF (net≥baz):');print(x0[x0.net>=BASE[0]['net']].sort_values('pf',ascending=False).head(5)[['f','p','net','pf','n','mdd','upi','OOS_upi']].round(3).to_string(index=False))
