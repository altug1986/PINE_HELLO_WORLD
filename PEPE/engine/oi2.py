exec(open('oi.py').read().split("\nrows=[dict(h='V10.1'")[0])
D=pd.read_pickle('oi1.pkl');pd.set_option('display.width',300);b=D.iloc[0]
h1=D[D.h=='H1 tükeniş'].copy();h1['n']=h1.p.str.extract(r'n(\d+)').astype(int);h1['P']=h1.p.str.extract(r'±(\d+)').astype(int);h1['O']=h1.p.str.extract(r'OI-(\d+)').astype(int)
print('H1 net farkı (n=6, satır fiyat %, sütun OI düşüş %):');print((h1[h1.n==6].pivot_table(index='P',columns='O',values='net')-b.net).round(0))
print('H1 net farkı n=12:');print((h1[h1.n==12].pivot_table(index='P',columns='O',values='net')-b.net).round(0))
print('H1 OOS Calmar n=6:');print(h1[h1.n==6].pivot_table(index='P',columns='O',values='Ocal').round(2))
print('H4 en yüksek net 5:');print(D[D.h=='H4 likidasyon'].sort_values('net',ascending=False).head(5)[['p','net','pf','mdd','q','y23','y24','y25','y26','Iq','Onet','Ocal']].round(2).to_string(index=False))
# birleşim H1 + H4
dp6,do6=chg(C,6),chg(OI,6)
ex=np.where((dp6>=3)&(do6<=-2),1,np.where((dp6<=-3)&(do6<=-2),-1,0)).astype(np.int64)
pl=vstate(C,(dp6<=-10)&(do6<=-5),(dp6>=10)&(do6<=-5),0.,.08,72,R7>60,R7<40,0).astype(np.int64)
rows=[dict(h='V10.1',**H(RUN())),dict(h='H1',**H(RUN(ex=ex,useex=1))),dict(h='H1+H4',**H(RUN(t=np.where(pl!=0,pl,t10).astype(np.int64),ex=ex,useex=1)))]
print(pd.DataFrame(rows)[['h','net','pct','pf','n','mdd','q','sharpe','uw','y23','y24','y25','y26','l12','r2_12','l3m','Iq','Onet','Opf','Ocal']].round(2).to_string(index=False))
