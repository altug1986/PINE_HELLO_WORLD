"""KAHİN — ileri yön olasılığı (yürüyen pencere, geleceği görmez) → SL/iz YOK, giriş-çıkış yalnız tahminden.

Özellikler (1 dk, her biri bar kapanışında bilinen): çok ufuklu getiri/oynaklık, RSI, EMA/BB sapması, OI değişimleri,
basis z, funding, top trader / genel L/S, balina net akışı, taker akışı, emir defteri dengesi, BTC/XMR getirisi ve artık,
4 sa ADX/DI. Saat/gün özelliği YOK.
Hedef: sonraki H dk getirisi > 0. Model: HistGradientBoosting (sığ), 6 aylık yürüyen yeniden eğitim (genişleyen geçmiş),
eğitim ile tahmin arasında H dk boşluk (etiket sızıntısı yok).
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import z01, z02, regime
from zec_data import load


def features(B):
    """Bellek dostu: her özellik doğrudan float32 matrise yazılır."""
    import gc
    n = len(B); names = []; cols = []
    def put(name, arr):
        names.append(name); cols.append(np.asarray(arr, dtype=np.float32))
    c = B.close.values.astype(float); h = B.high.values.astype(float); l = B.low.values.astype(float); v = B.volume.values.astype(float)
    sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]
    roll = lambda x, w, f: getattr(pd.Series(x).rolling(w, min_periods=max(2, w // 3)), f)().values
    lc = np.log(c); r1 = np.r_[0, np.diff(lc)]; vol60 = roll(r1, 60, 'std'); vol1d = roll(r1, 1440, 'std')
    for k in (1, 5, 15, 60, 240, 1440):
        put(f'ret{k}', (lc - sh(lc, k)) / (vol1d * np.sqrt(k)))
    put('volRatio', vol60 / vol1d)
    c1 = sh(c, 1); tr = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1))); atr = z01.rma(np.nan_to_num(tr), 60); del tr, c1
    put('atrP', atr / c * 100)
    d = np.diff(c, prepend=c[0]); g = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / 28, adjust=False).mean(); ls = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / 28, adjust=False).mean()
    put('rsi', (100 - 100 / (1 + g / ls)).values); del g, ls, d
    for k in (120, 480, 2880):
        put(f'zema{k}', (c - z02.ema(c, k)) / atr)
    mu = roll(c, 240, 'mean'); sd = roll(c, 240, 'std'); put('bbPos', (c - mu) / sd); del mu, sd
    put('hiDist', (c / roll(h, 1440, 'max') - 1) * 100); put('loDist', (c / roll(l, 1440, 'min') - 1) * 100)
    oi = np.log(B.oi_zec.values.astype(float))
    for k in (15, 60, 240, 1440):
        put(f'oi{k}', (oi - sh(oi, k)) * 100)
    put('oiZ7g', (oi - roll(oi, 10080, 'mean')) / roll(oi, 10080, 'std'))
    bs = B.basis_bps.values.astype(float); put('basis', bs); put('basisZ', (bs - roll(bs, 1440, 'mean')) / roll(bs, 1440, 'std'))
    put('funding', B.funding_pct.values)
    tp = np.log(B.tt_pos_ls.values.astype(float)); ta = np.log(B.tt_acc_ls.values.astype(float)); gl = np.log(B.glb_ls.values.astype(float))
    put('ttPos', tp); put('ttPos240', tp - sh(tp, 240)); put('ttPos1440', tp - sh(tp, 1440)); put('ttMinusGlb', tp - gl); put('ttPosMinusAcc', tp - ta)
    put('glbZ', (gl - roll(gl, 10080, 'mean')) / roll(gl, 10080, 'std')); del tp, ta, gl
    qv = B.quote_volume.values.astype(float); wn = (B.wh_buy_q50k.values - B.wh_sell_q50k.values).astype(float)
    for k in (10, 60, 240):
        put(f'whale{k}', roll(wn, k, 'sum') / roll(qv, k, 'sum'))
    put('bigPrint60', roll((B.wh_buy_q200k.values - B.wh_sell_q200k.values).astype(float), 60, 'sum') / roll(qv, 60, 'sum'))
    td = B.taker_delta_base.values.astype(float)
    for k in (5, 30, 240):
        put(f'taker{k}', roll(td, k, 'sum') / roll(v, k, 'sum'))
    # R4 — EMİLİM: akışın üretmesi beklenen fiyat hareketine göre gerçek hareket (geçmişle kalibre, geleceği görmez)
    fn = np.nan_to_num(td / np.where(v > 0, v, np.nan)); wf = np.nan_to_num(wn / np.where(qv > 0, qv, np.nan))
    for lab, f_ in (('akış', fn), ('balina', wf)):
        beta = sh(roll(r1 * f_, 1440, 'mean') / roll(f_ * f_, 1440, 'mean'), 1)        # dünkü etki katsayısı
        res = r1 - np.nan_to_num(beta) * f_
        for k in (5, 15, 30, 60):
            put(f'emilim_{lab}{k}', roll(res, k, 'sum') / (vol1d * np.sqrt(k)))
        put(f'etki_{lab}', beta * 1e4)
    # R4 — OI tuzağı: fiyat yatay, OI artıyor, akış bir yönde
    dp15 = (lc - sh(lc, 15)) / (vol1d * np.sqrt(15)); f15 = roll(td, 15, 'sum') / roll(v, 15, 'sum'); o15 = (oi - sh(oi, 15)) * 100
    put('oiTuzak', np.where(np.abs(dp15) < 0.5, o15 * np.sign(np.nan_to_num(f15)), 0.0))
    put('oiFiyatUyum', o15 * np.sign(np.nan_to_num(dp15)))
    bm1 = B.book_m1.values.astype(float); bp1 = B.book_p1.values.astype(float); put('book1', (bm1 - bp1) / (bm1 + bp1))
    bm5 = B.book_m5.values.astype(float); bp5 = B.book_p5.values.astype(float); put('book5', (bm5 - bp5) / (bm5 + bp5))
    bt = np.log(B.btc_close.values.astype(float)); xm = np.log(B.xmr_close.values.astype(float))
    for k in (5, 60, 240):
        put(f'btc{k}', (bt - sh(bt, k)) * 100); put(f'xmr{k}', (xm - sh(xm, k)) * 100)
    db = np.r_[0, np.diff(bt)]; beta = roll(r1 * db, 1440, 'mean') / roll(db * db, 1440, 'mean')
    put('resid60', roll(r1 - np.nan_to_num(beta) * db, 60, 'sum') / (vol1d * np.sqrt(60)))
    adx, di = regime.regime(B, '4h'); put('adx4h', adx); put('di4h', di); put('adxEğim', adx - sh(adx, 240))
    gc.collect()
    M = np.empty((n, len(cols)), np.float32)
    for j, col in enumerate(cols):
        M[:, j] = col
    del cols; gc.collect()
    M[~np.isfinite(M)] = np.nan
    return pd.DataFrame(M, index=B.index, columns=names, copy=False), atr


def walk_forward(B, Xf, H=120, step='182D', start='2022-01-01', sub=5, seed=0):
    c = B.close.values.astype(float); y = np.r_[c[H:] / c[:-H] - 1, np.full(H, np.nan)]
    p = np.full(len(c), np.nan); idx = B.index; t0 = pd.Timestamp(start)
    edges = list(pd.date_range(t0, idx[-1] + pd.Timedelta(step), freq=step))
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.where((idx < a - pd.Timedelta(minutes=H)) & ~np.isnan(y))[0][::sub]
        te = np.where((idx >= a) & (idx < b))[0]
        if len(te) == 0: continue
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, min_samples_leaf=500, l2_regularization=1.0, random_state=seed)
        Xtr = Xf.values[tr]; keep = np.where(np.isfinite(Xtr).mean(0) > 0.02)[0]            # eğitimde boş kolonları at
        m.fit(Xtr[:, keep], (y[tr] > 0).astype(int)); del Xtr
        p[te] = m.predict_proba(Xf.values[te][:, keep])[:, 1]
    return p, y
