import { createReadStream, createWriteStream } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { finished } from 'node:stream/promises';
import { loadRuntime } from './pine-runtime.mjs';

const [manifestPath, outputPath, summaryPath] = process.argv.slice(2);
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const missing = new Set();
const requests = new Map();
const chartStep = Number(manifest.main.timeframe) * 60000;
const series = new Map();
let requiresVolume = false;
const normalize = tf => {
    const value = String(tf).toUpperCase();
    if (value === 'D' || value === '1D') return '1440';
    return String(Number(value));
};

async function loadBars(file) {
    const bars = [];
    const lines = createInterface({ input: createReadStream(file), crlfDelay: Infinity });
    for await (const line of lines) {
        const [openTime, closeTime, open, high, low, close, volume] = line.split(',').map(Number);
        bars.push({ openTime, closeTime, open, high, low, close, volume });
    }
    return bars;
}

function lowerBound(bars, time) {
    let low = 0;
    let high = bars.length;
    while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (bars[mid].openTime < time) low = mid + 1;
        else high = mid;
    }
    return low;
}

function aggregate(bars, sourceMs, targetMs) {
    const result = [];
    let bar;
    let count = 0;
    for (const candle of bars) {
        const openTime = Math.floor(candle.openTime / targetMs) * targetMs;
        if (bar?.openTime !== openTime) {
            if (count * sourceMs === targetMs) result.push(bar);
            bar = { ...candle, openTime, closeTime: openTime + targetMs };
            count = 1;
        } else {
            bar.high = Math.max(bar.high, candle.high);
            bar.low = Math.min(bar.low, candle.low);
            bar.close = candle.close;
            bar.volume += candle.volume;
            count++;
        }
    }
    if (count * sourceMs === targetMs) result.push(bar);
    return result;
}

function resolveTicker(id) {
    if (Object.hasOwn(manifest.symbols, id) || id.includes(':')) return id;
    const matches = Object.keys(manifest.symbols).filter(symbol => manifest.symbols[symbol].ticker === id);
    return matches.length === 1 ? matches[0] : id;
}

const provider = {
    configure() {},
    async getSymbolInfo(id) {
        id = resolveTicker(id);
        const info = manifest.symbols[id];
        if (!info) {
            missing.add(id);
            throw new Error(`Sembol bilgisi veya OHLC eksik: ${id}`);
        }
        return info;
    },
    async getMarketData(id, timeframe, limit, start, end) {
        id = resolveTicker(id);
        const tf = normalize(timeframe);
        const key = `${id}@${tf}`;
        const isMain = id === manifest.main.symbol && tf === manifest.main.timeframe && start === manifest.start;
        if (!series.has(key)) {
            const feeds = manifest.feeds[id] ?? {};
            const source = Object.keys(feeds).map(Number).filter(n => Number(tf) % n === 0).sort((a, b) => b - a)[0];
            if (!source || Number(tf) < 1 || Number(tf) > 1440) {
                missing.add(`${id}@${timeframe}`);
                return [];
            }
            if (requiresVolume && !manifest.volume?.[id]?.[source]) {
                missing.add(`${key} (hacim)`);
                return [];
            }
            const raw = await loadBars(feeds[source]);
            series.set(key, Number(tf) === source ? raw : aggregate(raw, source * 60000, Number(tf) * 60000));
        }
        const bars = series.get(key);
        const lo = lowerBound(bars, start ?? -Infinity);
        const hi = lowerBound(bars, end === undefined ? Infinity : end + 1);
        const selected = bars.slice(start == null && limit ? Math.max(lo, hi - limit) : lo, hi);
        if (!selected.length) missing.add(`${id}@${timeframe}`);
        if (!isMain && (bars[0]?.openTime > manifest.start || bars.at(-1)?.closeTime < Math.floor((manifest.end + chartStep) / (Number(tf) * 60000)) * Number(tf) * 60000)) {
            missing.add(`${id}@${timeframe} (tarih kapsamı)`);
        }
        requests.set(key, { bars: bars.length, first: bars[0]?.openTime, last: bars.at(-1)?.openTime });
        return selected;
    },
};

async function test() {
    const { Indicator, PineTS } = await loadRuntime();
    const original = (await readFile(manifest.pine, 'utf8')).replace(/^\uFEFF/, '');
    const source = original.replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\/\/[^\n]*/g, '');
    if (!/^\s*\/\/@version\s*=\s*[56]\s*$/m.test(original)) throw new Error('Pine v5 veya v6 sürüm satırı gerekli.');
    requiresVolume = /\b(volume|vwap|vwma|obv|pvt|nvi|pvi|wad|wvad|mfi|iii|accdist)\b/.test(source);
    if (original.includes('__CYBORG_AUDIT_EQUITY_650D')) throw new Error('Denetim plot adı Pine kodunda kullanılıyor.');
    if (/^\s*import\s/m.test(source)) throw new Error('Pine import kütüphaneleri bu sürümde desteklenmiyor.');
    const unsupported = [...source.matchAll(/\brequest\s*\.\s*(\w+)\s*\(/g)].map(m => m[1]).filter(name => !['security', 'security_lower_tf'].includes(name));
    if (unsupported.length) throw new Error(`Desteklenmeyen Pine veri çağrısı: ${unsupported.join(', ')}`);
    const auditPlot = '__CYBORG_AUDIT_EQUITY_650D';
    const script = new Indicator(`${original}\nplot(strategy.equity, "${auditPlot}")\n`);
    if (script.getDeclarationType() !== 'strategy') throw new Error('Pine dosyası strategy() bildirmeli.');
    for (const [name, value] of Object.entries(manifest.inputs)) script.input[name] = value;
    for (const [name, value] of Object.entries(manifest.properties)) script.prop[name] = value;
    if (script.prop.use_bar_magnifier || (manifest.properties.use_bar_magnifier === undefined && /\buse_bar_magnifier\s*=\s*(?!false\b)\S+/.test(source))) {
        throw new Error('TradingView Bar Magnifier bu motor sürümünde desteklenmiyor.');
    }
    const main = await provider.getMarketData(manifest.main.symbol, manifest.main.timeframe, undefined, manifest.start, manifest.end);
    if (missing.size) throw new Error(`Eksik OHLC: ${[...missing].join(', ')}`);
    if (!main.length || main[0].openTime !== manifest.start || main.at(-1).openTime !== manifest.end) {
        throw new Error('Ana OHLC, başlangıç ve bitişteki bar açılışlarını kapsamıyor.');
    }
    if (main.some((bar, index) => index && main[index - 1].closeTime !== bar.openTime) && !manifest.allowGaps) {
        throw new Error('Ana OHLC aralığında eksik bar var. Önce veriyi doğrula.');
    }
    const chartTf = manifest.main.timeframe === '1440' ? 'D' : manifest.main.timeframe;
    const pine = new PineTS(provider, manifest.main.symbol, chartTf, main.length, manifest.start, manifest.end);
    pine.setTimezone(manifest.chartTimezone);
    const ctx = await pine.run(script);
    if (missing.size) throw new Error(`Eksik OHLC: ${[...missing].join(', ')}`);
    const state = ctx.strategy;
    const equity = ctx.plots[auditPlot]?.data;
    if (!state?.config || !equity || equity.length !== main.length || equity.some(p => !Number.isFinite(p.value))) {
        throw new Error('Strateji veya equity serisi eksik. Pine çalıştırması doğrulanamadı.');
    }
    const trades = state.closedtrades.map(t => ({
        side: t.size > 0 ? 'long' : 'short',
        entry_time: t.entry_time, exit_time: t.exit_time,
        entry_price: t.entry_price, exit_price: t.exit_price,
        qty: Math.abs(t.size), pnl: t.profit, commission: t.commission,
        entry_id: t.entry_id, exit_id: t.exit_id,
    }));
    if (trades.some(t => ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'qty', 'pnl', 'commission'].some(k => !Number.isFinite(t[k])))) {
        throw new Error('Emir günlüğünde geçersiz sayısal değer var.');
    }
    const stream = createWriteStream(outputPath);
    stream.write('bar_open_ms,equity\n');
    for (const point of equity) stream.write(`${point.time},${point.value}\n`);
    stream.end();
    await finished(stream);
    await writeFile(summaryPath, JSON.stringify({
        bars: main.length, feeds: Object.fromEntries(requests),
        config: state.config, initial_capital: state.initial_capital,
        pending_orders: state.pending_orders.length, open_legs: state.opentrades.length,
        end_open_pnl: state.openprofit, engine_net: state.netprofit,
        warnings: [...new Set((ctx.warnings ?? []).map(w => typeof w === 'string' ? w : JSON.stringify(w)))], trades,
    }));
}

try {
    await test();
} catch (error) {
    console.error(error.message);
    if (missing.size && !error.message.startsWith('Eksik OHLC')) console.error(`Eksik OHLC: ${[...missing].join(', ')}`);
    process.exitCode = 1;
}
