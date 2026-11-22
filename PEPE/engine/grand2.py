exec(open('grand.py').read().split("\nrnd=random.Random(7)")[0])
H1=dict(BC,mm=4.,md=3,mik=0,bflip=0,exo=0,po=3,psl=.08,pb=24,pre=60,slc=.02);h1=ev(H1);print('H1',{k:round(float(v),2) for k,v in h1.items()})
print('ABLASYON (H1 → tek ayar baza dönünce): net / PF / işlem / MaxDD')
for k in H1:
    if H1[k]!=BC[k]: c=dict(H1);c[k]=BC[k];m=ev(c);print(f"  {k}: {H1[k]}→{BC[k]} | {m['net']:.0f} / {m['pf']:.3f} / {m['n']} / {m['mdd']:.0f}")
def J3(m): return m['pf']-max(0,9200-m['net'])/500.-max(0,m['mdd']-285)/40.
best=[]
for st in [BC,H1,dict(H1,bflip=1),dict(H1,exo=1),dict(H1,mik=2)]:
    c=dict(st)
    for _ in range(3):
        for k,v in SP.items():
            bb=(J3(ev(c)),c[k])
            for val in v:
                c2=dict(c);c2[k]=val;j=J3(ev(c2))
                if j>bb[0]: bb=(j,val)
            c[k]=bb[1]
    best.append((J3(ev(c)),dict(c)))
print('değerlendirilen',len(CACHE),'süre',round(time.time()-t0),'s')
pd.set_option('display.width',400)
R=pd.DataFrame([dict(zip(SP,k),**m) for k,m in CACHE.items()])
ok=R[(R.net>=9200)&(R.mdd<=285)];print(f"\nnet≥9200 & MaxDD≤285: {len(ok)} | bunların max PF {ok.pf.max() if len(ok) else 0:.3f}")
ok2=R[(R.net>=9200)&(R.pf>=1.65)&(R.mdd<=300)];print(f"net≥9200 & PF≥1.65 & MaxDD≤300: {len(ok2)}")
print(R[(R.net>=9200)&(R.mdd<=285)].sort_values('pf',ascending=False).head(6)[list(SP)+['net','pf','n','Inet','Ipf','Onet','Opf','mdd','upi']].round(3).to_string(index=False))
R.to_pickle('grand2.pkl');import pickle;pickle.dump(best,open('grand2_best.pkl','wb'))
for j,c in best: print(round(j,3),{k:c[k] for k in SP if c[k]!=BC[k]},{k:round(float(v),2) for k,v in ev(c).items()})
