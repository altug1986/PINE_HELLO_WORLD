exec(open("ahf3.py").read().split("rows=[]")[0])
yr=B5.index.year
print('PEPE 1H ATR% medyan / yıl:',pd.Series(ATRP).groupby(yr).median().round(2).to_dict())
mA,mB,mM=mom(4),mom(48),mom(24)
a30,_=adx_htf(30);_,_,a5=adx5()
def lag(x,k): return np.concatenate([np.full(k,np.nan),x[:-k]])
gA=(a30>28)&(a5>20)&(a5>lag(a5,12));gB=(a30>25)&(a5>lag(a5,6));gM=(a30>18)&(a5>lag(a5,3))
for nm,s in [('A sinyal bar %',gA&(np.abs(mA)>=3)),('B sinyal bar %',gB&(np.abs(mB)>=2.5)),('M sinyal bar %',gM&(np.abs(mM)>=3))]:
    print(nm,(pd.Series(s).groupby(yr).mean()*100).round(2).to_dict())
src_=pd.Series(np.select([src==0,src==1,src==2,src==3,src==4],['RSIX','MIK','COL','B4H','BFLIP'],'MID'))
print('kaynak süre payı % / yıl:');print((pd.crosstab(yr,src_,normalize='index')*100).round(1).to_string())
V10=pickle.load(open('V10.pkl','rb'));te=V10[1];tp=V10[4]
s=pd.DataFrame({'y':B5.index[te].year,'src':src_.values[te],'pnl':tp})
print('motor neti / yıl:');print(s.pivot_table(index='y',columns='src',values='pnl',aggfunc='sum').round(0).to_string())
