exec(open('micro.py').read().split("\nrows=[dict(h='V11.1'")[0])
rows=[]
for zb,pb,Hh,sl in itertools.product([1.5,2,2.5,3,4],[0.4,0.5,0.6,0.8,1.0],[12,24,36,48],[.02,.03,.05]):
    ig=vstate(C,(oz1>=zb)&(bp1>=pb),(oz1>=zb)&(bp1<=-pb),0.,sl,Hh,F,F,0).astype(np.int64)
    r=R13(t=chain2(PL,ig,'colust'));h=HH(r);rows.append(dict(z=zb,pb=pb,H=Hh,sl=sl,trig=int((np.diff(np.r_[0,(ig!=0).astype(int)])==1).sum()),**h))
D=pd.DataFrame(rows);D.to_pickle('micro2.pkl');pd.set_option('display.width',260)
b=dict(net=8816.72,upi=31.52,IS_upi=46.97,OOS_upi=16.84)
print('UPI > V11.1 oranı',round((D.upi>b['upi']).mean(),2),'| net > V11.1',round((D.net>b['net']).mean(),2),'| OOS UPI > V11.1',round((D.OOS_upi>b['OOS_upi']).mean(),2))
for ax in ['z','pb','H','sl']: print(ax,D.groupby(ax)[['net','upi','IS_upi','OOS_upi','trig']].median().round(1).to_dict('index'))
print('UPI (satır z, sütun BTC fiyat %) — H24 sl3%:');print(D[(D.H==24)&(D.sl==.03)].pivot_table(index='z',columns='pb',values='upi').round(1))
print('OOS UPI — H24 sl3%:');print(D[(D.H==24)&(D.sl==.03)].pivot_table(index='z',columns='pb',values='OOS_upi').round(1))
D['Isel']=D.IS_upi;print('IS UPI ilk 8:');print(D.sort_values('Isel',ascending=False).head(8)[['z','pb','H','sl','trig','net','pf','n','mdd','ulcer','upi','IS_upi','OOS_upi','Ocal','y26','r2_12']].round(2).to_string(index=False))
