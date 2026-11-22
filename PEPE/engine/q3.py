exec(open('q2.py').read().split("mA=mom(4);mB=mom(48)")[0])
import pickle
S=pickle.load(open('S.pkl','rb'));X,_=pickle.load(open('X.pkl','rb'))
pA,pV=pos(S['A=V2']),pos(S['V3'])
for nm,r in X.items():
    px=pos(r); u=np.where((pA==pV)&(pV==px),pV,0); rec('UNAN',dict(x=nm),u)
    rec('W',dict(x=nm),np.sign(2*pA+2*pV+px))   # V4 + X tie-breaker when V2/V3 disagree
# multi-TF burst engines: 15m-equivalent and 1h-equivalent momentum on 5m bars
for n,m,A in [(12,4,25),(12,5,28),(24,5,25),(24,6,28)]:
    g=(a30>A)&(a5>lag(a5,6));mo=mom(n);px=P(g&(mo>=m),g&(mo<=-m))
    rec('UNAN',dict(x=f'MOM{n}/{m}/{A}'),np.where((pA==pV)&(pV==px),pV,0)); rec('W',dict(x=f'MOM{n}/{m}/{A}'),np.sign(2*pA+2*pV+px))
D=pd.DataFrame(rows);pd.set_option('display.width',250)
print(D.sort_values('calmar',ascending=False).head(10).round(2).to_string(index=False))
