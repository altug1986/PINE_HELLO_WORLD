exec(open('ptrail.py').read().split("def health")[0].split("x=pd.ExcelFile")[0]+open('ptrail.py').read().split("print('TV işlem'")[1].split("\n",1)[1].split("def health")[0])
LIB=pickle.load(open('/home/claude/data/LIB.pkl','rb'))
src=np.select([LIB['RSIX']!=0,LIB['MIK']!=0,LIB['COL']!=0,LIB['B4H']!=0,LIB['BFLIP']!=0],['RSIX','MIK','COL','B4H','BFLIP'],'MID')
h1,l1,c1,th1,_=htf(60);pc=np.r_[np.nan,c1[:-1]];tr1=np.fmax(np.fmax(h1-l1,np.abs(h1-pc)),np.abs(l1-pc));atr=np.nan_to_num(htf_map(rma(tr1,14)/c1*100,th1,3600),nan=2.)
vt=pd.qcut(atr[te],3,labels=['düşükVol','ortaVol','yüksekVol'])
out=[]
for act,gb in [(.06,.02),(.10,.03),(.15,.05),(.20,.05)]:
    _,dv,_=overlay(C,eq,te,tx,td,act,gb,0.3,TICK,COMM,CASH)
    s=pd.DataFrame({'dv':dv,'src':src[te],'vol':vt,'dir':np.where(td>0,'LONG','SHORT'),'oos':te>=I_SPLIT})
    for col in ['src','vol','dir']:
        g=s.groupby(col,observed=True).agg(top=('dv','sum'),IS=('dv',lambda v:v[~s.loc[v.index,'oos']].sum()),OOS=('dv',lambda v:v[s.loc[v.index,'oos']].sum()))
        for k,r in g.iterrows(): out.append(dict(ayar=f'{act:.0%}/{gb:.0%}',grup=f'{col}:{k}',fark=r.top,IS=r.IS,OOS=r.OOS))
O=pd.DataFrame(out);pd.set_option('display.width',200)
print(O.pivot_table(index='grup',columns='ayar',values='fark').round(0))
print('IS ve OOS ikisi de pozitif olan gruplar:');print(O[(O.IS>0)&(O.OOS>0)].round(0).to_string(index=False))
