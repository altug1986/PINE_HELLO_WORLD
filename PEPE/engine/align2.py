exec(open('micro.py').read().split("\nrows=[dict(h='V11.1'")[0])
def sh(x,k): return x if k==0 else np.r_[np.full(k,x[0]),x[:-k]]
print('V11.1 PEPE OI kaydırma duyarlılığı (pozitif = daha geç bilgi):')
for k in [0,1,2,3]:
    oi_=sh(OI,k);do=chg(oi_,6)
    ex=np.where((dp6>=3)&(do<=-2),1,np.where((dp6<=-3)&(do<=-2),-1,0)).astype(np.int64)
    pl=vstate(C,(dp6<=-4)&(do<=-4),(dp6>=4)&(do<=-4),0.,.05,48,R7>55,R7<45,0).astype(np.int64)
    h=HH(R13(t=chain2(pl),ex=ex));print(f'  {k:+d} bar: net {h["net"]:.0f} | UPI {h["upi"]:.1f} | OOS UPI {h["OOS_upi"]:.1f} | OOS Calmar {h["Ocal"]:.2f}')
h=HH(R13(t=chain2(np.zeros(N,np.int64)),ex=np.zeros(N,np.int64)));print(f'  OI tamamen yok (V10.1 eşdeğeri): net {h["net"]:.0f} | UPI {h["upi"]:.1f} | OOS UPI {h["OOS_upi"]:.1f}')
