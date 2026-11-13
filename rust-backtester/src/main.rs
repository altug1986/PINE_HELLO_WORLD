use std::{env,fs::{self,File},io::{BufRead,BufReader,BufWriter,Write},collections::HashMap,path::Path};
#[derive(Clone)]
struct C {ohlc:String,signals:String,out:String,initial:f64,cash:f64,fee:f64,ticks:f64,tick:f64,legs:usize,on_close:bool}
fn number(s:&str)->Result<f64,String>{let v:f64=s.parse().map_err(|_|format!("Invalid number: {s}"))?;if !v.is_finite(){return Err("Nonfinite number".into())}Ok(v)}
fn args()->Result<C,String>{
 let mut c=C{ohlc:String::new(),signals:String::new(),out:"results/run".into(),initial:2500.,cash:250.,fee:0.04,ticks:1.,tick:0.001,legs:2,on_close:false};
 let mut it=env::args().skip(1);
 while let Some(k)=it.next(){if k=="--on-close"{c.on_close=true;continue;}let v=it.next().ok_or(format!("Value missing: {k}"))?;
 match k.as_str(){"--ohlc"=>c.ohlc=v,"--signals"=>c.signals=v,"--out"=>c.out=v,"--initial"=>c.initial=number(&v)?,
 "--cash-per-leg"=>c.cash=number(&v)?,"--commission-percent"=>c.fee=number(&v)?,
 "--slippage-ticks"=>c.ticks=number(&v)?,"--tick-size"=>c.tick=number(&v)?,
 "--max-legs"=>c.legs=v.parse().map_err(|_|"Invalid legs".to_string())?,
 _=>return Err(format!("Unsupported flag: {k}"))}}
 if c.ohlc.is_empty()||c.signals.is_empty()||c.initial<=0.||c.cash<=0.||c.tick<=0.||c.legs==0||c.fee<0.||c.ticks<0.{return Err("Usage: --ohlc file.csv --signals file.csv [--out results/run] [--on-close] [--initial 2500] [--cash-per-leg 250] [--max-legs 2] [--commission-percent 0.04] [--slippage-ticks 1] [--tick-size .001]".into())}
 Ok(c)
}
#[derive(Clone,Copy)]
enum A{Long,Short,Add,Exit,Reduce(f64)}
fn signals(path:&str)->Result<HashMap<String,A>,String>{
 let mut lines=BufReader::new(File::open(path).map_err(|e|e.to_string())?).lines();
 if !lines.next().ok_or("Empty signals file")?.map_err(|e|e.to_string())?.trim_start_matches('\u{feff}').starts_with("timestamp,action"){return Err("Expected timestamp,action,fraction header".into())}
 let mut map=HashMap::new();
 for row in lines {let s=row.map_err(|e|e.to_string())?;if s.trim().is_empty(){continue}let p:Vec<_>=s.split(',').collect();if p.len()<2{return Err("Invalid signal row".into())}
 let a=match p[1].trim().to_uppercase().as_str(){"LONG"=>A::Long,"SHORT"=>A::Short,"ADD"=>A::Add,"EXIT"=>A::Exit,"REDUCE"=>{let f=number(p.get(2).copied().unwrap_or(""))?;if f<=0.||f>1.{return Err("REDUCE fraction must be (0,1]".into())}A::Reduce(f)},_=>return Err("Unknown action".into())};
 if map.insert(p[0].trim().to_string(),a).is_some(){return Err("Duplicate signal timestamp".into())}}
 Ok(map)
}
struct Bar {t:String,o:f64,h:f64,l:f64,c:f64}
fn bar(s:&str)->Result<Bar,String>{
 let p:Vec<_>=s.split(',').collect();if p.len()!=5{return Err(format!("Bad OHLC row: {s}"))}
 let (o,h,l,c)=(number(p[1].trim())?,number(p[2].trim())?,number(p[3].trim())?,number(p[4].trim())?);
 if l<=0.||l>o||l>c||h<o||h<c||h<l{return Err(format!("Invalid OHLC: {s}"))}
 Ok(Bar{t:p[0].trim().into(),o,h,l,c})
}
struct P{side:f64,qty:f64,avg:f64,legs:usize,fee:f64,entry:String}
struct T{entry:String,exit:String,side:&'static str,qty:f64,ep:f64,xp:f64,gross:f64,entry_fee:f64,exit_fee:f64,net:f64}
struct E{c:C,cash:f64,pos:Option<P>,trades:Vec<T>,peak:f64,maxdd:f64,maxddpct:f64,maxadverse:f64,last:f64,bars:usize}
impl E{
 fn new(c:C)->Self{Self{cash:c.initial,peak:c.initial,last:c.initial,c,pos:None,trades:Vec::new(),maxdd:0.,maxddpct:0.,maxadverse:0.,bars:0}}
 fn buy(&self,m:f64)->f64{m+self.c.tick*self.c.ticks}
 fn sell(&self,m:f64)->f64{m-self.c.tick*self.c.ticks}
 fn fee(&self,q:f64,p:f64)->f64{q*p*self.c.fee/100.}
 fn enter(&mut self,side:f64,m:f64,t:&str){
 let p=if side>0.{self.buy(m)}else{self.sell(m)};if p<=0.{return}
 let q=self.c.cash/p;let fee=self.fee(q,p);
 match self.pos.as_mut(){
 None=>{self.cash-=fee;self.pos=Some(P{side,qty:q,avg:p,legs:1,fee,entry:t.into()});},
 Some(x) if x.side==side&&x.legs<self.c.legs=>{self.cash-=fee;x.avg=(x.avg*x.qty+p*q)/(x.qty+q);x.qty+=q;x.fee+=fee;x.legs+=1;},
 _=>{}
 }}
 fn exit(&mut self,f:f64,m:f64,t:&str){
 let Some(mut x)=self.pos.take() else{return};
 let q=if f>=1.-1e-12{x.qty}else{x.qty*f};
 let p=if x.side>0.{self.sell(m)}else{self.buy(m)};if p<=0.{self.pos=Some(x);return}
 let ef=x.fee*q/x.qty;let xf=self.fee(q,p);let gross=(p-x.avg)*q*x.side;let net=gross-ef-xf;
 self.cash+=gross-xf;
 self.trades.push(T{entry:x.entry.clone(),exit:t.into(),side:if x.side>0.{"LONG"}else{"SHORT"},qty:q,ep:x.avg,xp:p,gross,entry_fee:ef,exit_fee:xf,net});
 x.qty-=q;x.fee-=ef;if x.qty>1e-10{self.pos=Some(x)}
 }
 fn act(&mut self,a:A,m:f64,t:&str){match a{
 A::Long=>match self.pos.as_ref().map(|p|p.side){None=>self.enter(1.,m,t),Some(x) if x<0.=>self.exit(1.,m,t),_=>{}},
 A::Short=>match self.pos.as_ref().map(|p|p.side){None=>self.enter(-1.,m,t),Some(x) if x>0.=>self.exit(1.,m,t),_=>{}},
 A::Add=>if let Some(side)=self.pos.as_ref().map(|p|p.side){self.enter(side,m,t)},
 A::Exit=>self.exit(1.,m,t),A::Reduce(f)=>self.exit(f,m,t)}}
 fn equity(&self,m:f64)->f64{self.cash+self.pos.as_ref().map_or(0.,|p|p.qty*p.side*(m-p.avg))}
 fn observe(&mut self,b:&Bar)->(f64,f64){
 let eq=self.equity(b.c);self.peak=self.peak.max(eq);let dd=(self.peak-eq).max(0.);self.maxdd=self.maxdd.max(dd);
 if self.peak>0.{self.maxddpct=self.maxddpct.max(dd*100./self.peak)}
 let adverse=match self.pos.as_ref().map(|p|p.side){Some(s) if s>0.=>b.l,Some(_)=>b.h,None=>b.c};
 self.maxadverse=self.maxadverse.max((self.peak-self.equity(adverse)).max(0.));self.last=eq;self.bars+=1;(eq,dd)
 }
}
fn run(c:C)->Result<(),String>{
 let mut actions=signals(&c.signals)?;let fi=File::open(&c.ohlc).map_err(|e|e.to_string())?;let mut lines=BufReader::new(fi).lines();
 let header=lines.next().ok_or("OHLC empty")?.map_err(|e|e.to_string())?;
 if header.trim_start_matches('\u{feff}').trim()!="timestamp,open,high,low,close"{return Err("Wrong OHLC header".into())}
 if let Some(parent)=Path::new(&c.out).parent(){if !parent.as_os_str().is_empty(){fs::create_dir_all(parent).map_err(|e|e.to_string())?}}
 let mut eqw=BufWriter::new(File::create(format!("{}_equity.csv",c.out)).map_err(|e|e.to_string())?);writeln!(eqw,"timestamp,equity,balance,drawdown_close").map_err(|e|e.to_string())?;
 let mut e=E::new(c.clone());let mut pending=None;let mut prev=String::new();
 for line in lines{let s=line.map_err(|e|e.to_string())?;if s.trim().is_empty(){continue}let b=bar(&s)?;
 if b.t.is_empty()||(!prev.is_empty()&&b.t<=prev){return Err("OHLC timestamps must increase".into())}
 if let Some(a)=pending.take(){e.act(a,b.o,&b.t)}
 if let Some(a)=actions.remove(&b.t){if c.on_close{e.act(a,b.c,&b.t)}else{pending=Some(a)}}
 let (eq,dd)=e.observe(&b);writeln!(eqw,"{},{:.10},{:.10},{:.10}",b.t,eq,e.cash,dd).map_err(|e|e.to_string())?;prev=b.t;
 }
 eqw.flush().map_err(|e|e.to_string())?;
 if e.bars==0{return Err("No bars".into())}
 let mut w=BufWriter::new(File::create(format!("{}_trades.csv",c.out)).map_err(|e|e.to_string())?);
 writeln!(w,"entry_time,exit_time,side,qty,entry_price,exit_price,gross_pnl,entry_fee,exit_fee,net_pnl").map_err(|e|e.to_string())?;
 let (mut grosswin,mut grossloss,mut wins,mut net)=(0.,0.,0usize,0.);
 for t in &e.trades{writeln!(w,"{},{},{},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10}",t.entry,t.exit,t.side,t.qty,t.ep,t.xp,t.gross,t.entry_fee,t.exit_fee,t.net).map_err(|e|e.to_string())?;
 net+=t.net;if t.net>0.{grosswin+=t.net;wins+=1}else{grossloss-=t.net}}
 w.flush().map_err(|e|e.to_string())?;
 let pf=if grossloss>0.{format!("{:.8}",grosswin/grossloss)}else{"null".into()};
 let wr=if !e.trades.is_empty(){format!("{:.8}",wins as f64*100./e.trades.len() as f64)}else{"null".into()};
 let json=format!("{{\n \"tv_parity_verified\":false,\n \"pine_interpreter\":false,\n \"bars\":{},\n \"closed_trades\":{},\n \"net_closed_pnl\":{:.8},\n \"net_including_open_mark\":{:.8},\n \"profit_factor\":{},\n \"win_rate_pct\":{},\n \"max_drawdown_close\":{:.8},\n \"max_drawdown_close_pct\":{:.8},\n \"max_drawdown_adverse_proxy\":{:.8},\n \"unmatched_signals\":{}\n}}\n",e.bars,e.trades.len(),net,e.last-c.initial,pf,wr,e.maxdd,e.maxddpct,e.maxadverse,actions.len());
 fs::write(format!("{}_summary.json",c.out),&json).map_err(|e|e.to_string())?;println!("{json}");Ok(())
}
fn main(){if let Err(e)=args().and_then(run){eprintln!("{e}");std::process::exit(2)}}
#[cfg(test)]
mod tests{use super::*;
 fn c()->C{C{ohlc:"".into(),signals:"".into(),out:"".into(),initial:1000.,cash:100.,fee:0.,ticks:0.,tick:0.01,legs:2,on_close:true}}
 #[test]fn long(){let mut e=E::new(c());e.act(A::Long,10.,"a");e.act(A::Exit,11.,"b");assert!((e.cash-1010.).abs()<1e-9)}
 #[test]fn short(){let mut e=E::new(c());e.act(A::Short,10.,"a");e.act(A::Exit,9.,"b");assert!((e.cash-1010.).abs()<1e-9)}
 #[test]fn reduce(){let mut e=E::new(c());e.act(A::Long,10.,"a");e.act(A::Reduce(0.1),11.,"b");assert!((e.pos.unwrap().qty-9.).abs()<1e-9)}
 #[test]fn no_flip(){let mut e=E::new(c());e.act(A::Short,10.,"a");e.act(A::Long,11.,"b");assert!(e.pos.is_none())}
}