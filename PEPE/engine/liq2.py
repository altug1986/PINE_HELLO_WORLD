exec(open('flow1.py').read().split("\nrows=[dict(h='V11'")[0])
def chainP(pl):
    t=np.zeros(N,np.int64)
    for k in ORDER[::-1]: t=np.where(ENG[k]!=0,ENG[k],t)
    return np.where(pl!=0,pl,t).astype(np.int64)
rows=[dict(p='V11',**HH(R13()))]
for P,O_,sl,hold,xr in itertools.product([4,5,6],[3,4],[.03,.05,.08],[12,24,48],[55,60]):
    pl=vstate(C,(dp6<=-P)&(do6<=-O_),(dp6>=P)&(do6<=-O_),0.,sl,hold,R7>xr,R7<100-xr,0).astype(np.int64)
    rows.append(dict(p=f'P{P} O{O_} sl{sl} h{hold} rsi{xr}',trig=int((np.diff(np.r_[0,(pl!=0).astype(int)])==1).sum()),**HH(R13(t=chainP(pl)))))
D=pd.DataFrame(rows);D.to_pickle('liq2.pkl');pd.set_option('display.width',300);b=D.iloc[0]
s=D[(D.upi>b.upi)&(D.net>=b.net)&(D.OOS_upi>=b.OOS_upi)&(D.IS_upi>=b.IS_upi*0.97)]
print(D.iloc[:1][['net','mdd','ulcer','upi','IS_upi','OOS_upi','Ocal']].round(2).to_string(index=False))
print('UPI↑ net↑ OOS UPI↑ geçen:',len(s),'/',len(D)-1);print(s.sort_values('upi',ascending=False).head(10)[['p','trig','net','pct','pf','n','mdd','ulcer','upi','IS_upi','OOS_upi','Ocal','y26','r2_12']].round(2).to_string(index=False))
