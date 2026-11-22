exec(open('joint2.py').read().split("\nJ4=dict(")[0])
J4=dict(k='J4',n=6,x=3,y=3,zb=1.5);J2=dict(k='J2',n=12,x=4,y=4,zb=1)
for s in [0,1,2]:
    b=runj(s,[])[0];m=runj(s,[J4,J2])[0]
    print(f"s{s}: ilk yarı {b['Inet']:.0f}→{m['Inet']:.0f} | ikinci yarı {b['Onet']:.0f}→{m['Onet']:.0f} | PF {b['pf']:.3f}→{m['pf']:.3f} | IS/OOS UPI {b['Iupi']:.1f}/{b['Oupi']:.1f}→{m['Iupi']:.1f}/{m['Oupi']:.1f} | işlem {b['n']}→{m['n']}")
