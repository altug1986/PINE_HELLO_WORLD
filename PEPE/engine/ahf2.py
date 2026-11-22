exec(open("ahf.py").read().split("rows=[]")[0])
D=pd.read_pickle('ahf.pkl');pd.set_option('display.width',260)
print('eşik bazında en iyi (IS seçimi):')
D['Iq']=D.Inet/D.Imdd;D['Isel']=D.Inet/1000*np.sqrt(D.Iq)
print(D.sort_values('Isel',ascending=False).groupby(['kind','v']).head(1).round(2).to_string(index=False))
print('komşu kontrolü atr4 tr1 (slc x mc), rb0, tpc0:')
print(D[(D.kind=='atr')&(D.v==4)&(D.tr==1)&(D.rb==0)&(D.tpc==0)].pivot_table(index='slc',columns='mc',values='net').round(0))
print(D[(D.kind=='atr')&(D.v==4)&(D.tr==1)&(D.rb==0)&(D.tpc==0)].pivot_table(index='slc',columns='mc',values='Ocal').round(2))
print('komşu: eşik (slc .05 mc288 rb0 tpc0 tr1):');print(D[(D.slc==.05)&(D.mc==288)&(D.rb==0)&(D.tpc==0)&(D.tr==1)][['kind','v','net','pf','n','mdd','q','Ipf','Opf','Ocal','flips']].round(2).to_string(index=False))
# MAE for key configs
def mm(r):
    m=mae(r);return np.percentile(m,[95,99]).round(2).tolist()+[round(m.max(),2),int((m>8).sum()),int((m>10).sum())]
V9e=run_ahf(C,t8,R7,np.zeros(N),0.,1.,1,0,1,90.,60.,12,.03,TICK,COMM,CASH)
print('V9 (motor) MAE P95/P99/max/>8/>10:',mm(V9e),'net',round(V9e[0][-1]))
for slc,mc,rb in [(.05,288,0),(.05,288,1),(.03,288,0),(.05,144,0)]:
    r=run_ahf(C,t8,R7,THR('atr',4),0.,slc,mc,rb,1,90.,60.,12,.03,TICK,COMM,CASH);print(f'AHF atr4 slc{slc} mc{mc} rb{rb}: net {r[0][-1]:.0f} MAE',mm(r))
