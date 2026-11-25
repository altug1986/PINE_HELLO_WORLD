import json
T = json.load(open('/home/claude/zecwork/tv_tablo.json'))
comps = [('P04', 'P04_htf4h'), ('P08', 'P08_zema'), ('P09', 'P09_spike'), ('P10', 'P10_bb'), ('P15', 'P15_resid')]
masks = '\n'.join(f'string M_{s}_L = "{T[k + "_L"]}"\nstring M_{s}_S = "{T[k + "_S"]}"' for s, k in comps)
SOK = 'BINANCE_ZECUSDT_ZEC-1M-CYBORG_1M_BOTKODU'
pine = f'''//@version=6
// ═══════════════════════════════════════════════════════════════════════════════════════════════
// CYBORG | ZEC 1M DENGELİ-TV V1.0  —  BINANCE:ZECUSDT.P, 1 dakika
// REV-01 (2026-11-25): ilk sürüm. Araştırma modeli "DENGELİ"nin TradingView'de hesaplanabilen çekirdeği.
//   Oylar (5): P04 4s trend · P08 EMA480 sapması · P09 sıçrama dönüşü · P10 Bollinger 240 · P15 BTC artığı
//   (P07 RSI: ilk yarı seçiminde hiçbir hücrede seçilmedi → oy vermez, kodda yok)
//   Araştırma modelindeki top trader / balina / emir defteri / basis / OI / global L-S oyları TV'de olmadığı için YOK.
//   Rejim (96 hücre): 4s ADX(14) yönü × BTC getiri kor.(72s, 90g medyan) × ZEC göreli ATR% (BTC/XMR/BAT/PEPE, 720s z)
//                     × ATR korelasyonu (168s, 90g medyan) × günlük EMA20 × XMR 24s yönü  — hepsi kapanmış bardan.
//   Kural tablosu ZEC 2021-01→2023-12 (ilk yarı) verisinden sabitlendi; burada yeniden seçim YOK.
//   Giriş: oy ≥2 + fitil (son 10 dk mumda yön fitili ≥ %40) · long dikkat: XMR 24s ≥ +%4 veya BAT yükselişte → long için 4 oy
//   Çıkış: ters oy · SL 28×ATR(60) · 3.840 dk aynı yönde oy gelmezse · sıçrama (10 dk ≥ +%3) short'u kapatır, 60 dk short yok
// ARAŞTIRMA MOTORU (1 dk OHLC, %0,04×2 + 1 tick, sabit 500 $): net 5.665 $ (%1.133) · 559 işlem · PF 1,89
//   MaxDD 423 · DD P95/P99 297/364 (30 dk MTM) · Sharpe 1,92 · Calmar 2,28 · ilk/ikinci yarı 2.897/2.769 · Kasım 2026 +81
//   TV'de DOĞRULANMADI — ilk TV export'u ile işlem işlem karşılaştırılacak.
// ═══════════════════════════════════════════════════════════════════════════════════════════════
strategy("CYBORG | ZEC 1M DENGELİ-TV V1.0", shorttitle="ZEC1M-DTV", overlay=true, initial_capital=500,
     default_qty_type=strategy.cash, default_qty_value=500, commission_type=strategy.commission.percent, commission_value=0.04,
     slippage=1, process_orders_on_close=true, pyramiding=0, margin_long=0, margin_short=0, max_bars_back=2000)

// ── Sabitler ────────────────────────────────────────────────────────────────────────────────────
string BTC = "BINANCE:BTCUSDT.P"
string XMR = "BINANCE:XMRUSDT.P"
string BAT = "BINANCE:BATUSDT.P"
string PEP = "BINANCE:1000PEPEUSDT.P"
int   K_L = 2, K_S = 2, K_DIKKAT = 2
float TS = 28.0
int   TH = 3840
float WF = 0.4
float R2 = math.sqrt(2.0)

// Kural tablosu: karakter i = rejim hücresi i (0–95) için oy açık mı (1) — ZEC ilk yarı seçimi, sabit
{masks}

// ── 1 dk seriler ──────────────────────────────────────────────────────────────────────────────
float atr  = ta.rma(ta.tr(true), 60)
float btc1 = request.security(BTC, "1", close)
float xmr1 = request.security(XMR, "1", close)
float bat1 = request.security(BAT, "1", close)

// ── Oylar ─────────────────────────────────────────────────────────────────────────────────────
float h4d = request.security(syminfo.tickerid, "240", (close - ta.ema(close, 50))[1], lookahead = barmerge.lookahead_on)
int vP04 = na(h4d) ? 0 : int(math.sign(h4d))
float zE = nz((close - ta.ema(close, 480)) / atr)
int vP08 = zE <= -8 * R2 ? 1 : zE >= 8 * R2 ? -1 : 0
float m20 = nz((close - close[40]) / atr)
float upF = nz(ta.highest(m20 >= 10 * R2 ? 1.0 : 0.0, 120))
float dnF = nz(ta.highest(m20 <= -10 * R2 ? 1.0 : 0.0, 120))
int vP09 = int(dnF - upF)
float mu = ta.sma(close, 240)
float sd = ta.stdev(close, 240, false)
int vP10 = close < mu - 2.5 * sd ? 1 : close > mu + 2.5 * sd ? -1 : 0
float dz = nz(math.log(close / close[1]))
float db = nz(math.log(btc1 / btc1[1]))
float beta = nz(ta.sma(dz * db, 1440) / ta.sma(db * db, 1440))
float eR = dz - beta * db
float q = nz(math.sum(eR, 60) / (ta.stdev(eR, 1440, false) * math.sqrt(60)))
int vP15 = q > 2 ? 1 : q < -2 ? -1 : 0

// ── Rejim: 4s ADX ─────────────────────────────────────────────────────────────────────────────
dmi4() =>
    [p, m, a] = ta.dmi(14, 14)
    [a[1], p[1], m[1]]
[adx4, pdi4, mdi4] = request.security(syminfo.tickerid, "240", dmi4(), lookahead = barmerge.lookahead_on)
int RG = na(adx4) ? 9 : adx4 >= 20 ? int(math.sign(pdi4 - mdi4)) : 0
bool deB = request.security(syminfo.tickerid, "D", (close > ta.ema(close, 20))[1], lookahead = barmerge.lookahead_on)
int De = deB ? 1 : 0
int Xb = nz(xmr1[1] / xmr1[1441]) > 1 ? 1 : 0

// ── Rejim: saatlik akran özellikleri (kapanmış 1 s bar, saat başında güncellenir) ────────────────
atrp() => ta.rma(ta.tr(true), 14) / close * 100
lret() => math.log(close / close[1])
[zA, zR] = request.security(syminfo.tickerid, "60", [atrp()[1], lret()[1]], lookahead = barmerge.lookahead_on)
[bA, bR] = request.security(BTC, "60", [atrp()[1], lret()[1]], lookahead = barmerge.lookahead_on)
float xA = request.security(XMR, "60", atrp()[1], lookahead = barmerge.lookahead_on)
float tA = request.security(BAT, "60", atrp()[1], lookahead = barmerge.lookahead_on)
float pA = request.security(PEP, "60", atrp()[1], lookahead = barmerge.lookahead_on)

lastVals(array<float> a, int n) =>
    array<float> t = array.new_float(0)
    int sz = array.size(a)
    if sz > 0
        for i = math.max(0, sz - n) to sz - 1
            float v = array.get(a, i)
            if not na(v)
                array.push(t, v)
    t
medLast(array<float> a, int n, int minp) =>
    array<float> t = lastVals(a, n)
    array.size(t) >= minp ? array.median(t) : na
zLast(array<float> a, int n, int minp) =>
    array<float> t = lastVals(a, n)
    float x = array.size(a) > 0 ? array.get(a, array.size(a) - 1) : na
    array.size(t) >= minp and not na(x) ? (x - array.avg(t)) / array.stdev(t, false) : na
corrLast(array<float> a, array<float> b, int n, int minp) =>
    int sz = array.size(a)
    float sx = 0.0, float sy = 0.0, float sxx = 0.0, float syy = 0.0, float sxy = 0.0, int k = 0
    if sz > 0
        for i = math.max(0, sz - n) to sz - 1
            float x = array.get(a, i)
            float y = array.get(b, i)
            if not na(x) and not na(y)
                sx += x, sy += y, sxx += x * x, syy += y * y, sxy += x * y, k += 1
    float vx = sxx - sx * sx / k
    float vy = syy - sy * sy / k
    k >= minp and vx > 0 and vy > 0 ? (sxy - sx * sy / k) / math.sqrt(vx * vy) : na
pushCap(array<float> a, float v) =>
    array.push(a, v)
    if array.size(a) > 2200
        array.shift(a)

var array<float> aZA = array.new_float(0)
var array<float> aPM = array.new_float(0)
var array<float> aZR = array.new_float(0)
var array<float> aBR = array.new_float(0)
var array<float> aREL = array.new_float(0)
var array<float> aBC = array.new_float(0)
var array<float> aAC = array.new_float(0)
var int K = 1
var int Vh = 1
var int A = 1
bool newH = nz(ta.change(time("60"))) != 0
if newH
    array<float> pk = array.new_float(0)
    if not na(bA)
        array.push(pk, bA)
    if not na(xA)
        array.push(pk, xA)
    if not na(tA)
        array.push(pk, tA)
    if not na(pA)
        array.push(pk, pA)
    float pm = array.size(pk) > 0 ? array.median(pk) : na
    pushCap(aZA, zA), pushCap(aPM, pm), pushCap(aZR, zR), pushCap(aBR, bR)
    pushCap(aREL, (not na(zA) and not na(pm) and pm > 0) ? math.log(zA / pm) : na)
    float relz = zLast(aREL, 720, 240)
    float atrC = corrLast(aZA, aPM, 168, 96)
    float btcC = corrLast(aZR, aBR, 72, 48)
    pushCap(aBC, btcC), pushCap(aAC, atrC)
    K  := nz(btcC - medLast(aBC, 2160, 500)) >= 0 ? 1 : 0
    Vh := nz(relz) >= 0 ? 1 : 0
    A  := nz(atrC - medLast(aAC, 2160, 500)) >= 0 ? 1 : 0

int g  = RG == 9 ? -1 : RG + 1
int rc = g < 0 ? -1 : ((g * 8 + K * 4 + Vh * 2 + A) * 2 + De) * 2 + Xb
on(string m, int v, int side) => rc >= 0 and v == side and str.substring(m, rc, rc + 1) == "1" ? 1 : 0
int L = on(M_P04_L, vP04, 1) + on(M_P08_L, vP08, 1) + on(M_P09_L, vP09, 1) + on(M_P10_L, vP10, 1) + on(M_P15_L, vP15, 1)
int S = on(M_P04_S, vP04, -1) + on(M_P08_S, vP08, -1) + on(M_P09_S, vP09, -1) + on(M_P10_S, vP10, -1) + on(M_P15_S, vP15, -1)

// ── Giriş kalitesi: fitil + dikkat ─────────────────────────────────────────────────────────────
float Hw = ta.highest(high, 10)
float Lw = ta.lowest(low, 10)
float Ow = open[9]
float rw = math.max(Hw - Lw, 1e-9)
float wkL = (math.min(Ow, close) - Lw) / rw
float wkS = (Hw - math.max(Ow, close)) / rw
bool batUp = bat1 > ta.ema(bat1, 1440) and nz(bat1 / bat1[1440]) > 1
bool dikkatL = nz(xmr1 / xmr1[1440] - 1) >= 0.04 or batUp
bool eL = L >= K_L + (dikkatL ? K_DIKKAT : 0) and wkL >= WF
bool eS = S >= K_S and wkS >= WF

// ── Yön, çıkış, giriş (araştırma motoru sim11 ile aynı sıra) ────────────────────────────────────
var int ti = 0
var int noS = -1
bool spk = nz(close / close[10] - 1) >= 0.03
if spk
    noS := bar_index + 60
bool aL = L >= K_L
bool aS = S >= K_S and not aL
int wA = aL ? 1 : (aS and bar_index > noS) ? -1 : 0
int pos = strategy.position_size > 0 ? 1 : strategy.position_size < 0 ? -1 : 0
bool guard = pos == -1 and spk
bool tout = pos != 0 and bar_index - ti >= TH
bool ex = pos != 0 and (guard or (wA != 0 and wA != pos) or tout)
int blk = tout and wA == pos ? pos : 0
if ex
    strategy.close_all(comment = "EXIT-ALL_{SOK}")
    pos := 0
if pos != 0 and wA == pos
    ti := bar_index
if pos == 0 and wA != 0 and wA != blk and not na(atr) and ((wA == 1 and eL) or (wA == -1 and eS))
    float ep = close + syminfo.mintick * wA
    ti := bar_index
    if wA == 1
        strategy.entry("L", strategy.long, comment = "ENTER-LONG_{SOK}")
        strategy.exit("SL-L", from_entry = "L", stop = ep - TS * atr, comment = "EXIT-ALL_{SOK}")
    else
        strategy.entry("S", strategy.short, comment = "ENTER-SHORT_{SOK}")
        strategy.exit("SL-S", from_entry = "S", stop = ep + TS * atr, comment = "EXIT-ALL_{SOK}")

plot(L, "Long oy", color.green, display = display.data_window)
plot(S, "Short oy", color.red, display = display.data_window)
plot(rc, "Rejim hücresi", color.gray, display = display.data_window)
'''
open('/home/claude/pine_hello_world/ZEC/pine/CYBORG_ZEC_1M_DENGELI_TV_V1.0.pine', 'w').write(pine); print(len(pine.splitlines()), 'satır')
