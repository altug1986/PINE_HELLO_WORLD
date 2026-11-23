import React, { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";

type Bar = { time: string; timeMs: number; open: number; high: number; low: number; close: number };
type Key = "rsiTrig" | "adxNow" | "adx15" | "adx10" | "sl" | "arm" | "pull" | "rev" | "sun";
type RangeDef = { key: Key; label: string; min: number; max: number; step: number; kind: "int" | "float" };
type Combo = { rsiTrig: number; adxNow: number; adx15: number; adx10: number; sl: number; arm: number; pull: number; rev: number; sun: number };
type Row = Combo & {
  id: string;
  score: number;
  netProfitPct: number;
  openPnlPct: number;
  maxDrawdownPct: number;
  tradeCount: number;
  winRatePct: number;
  profitFactor: number;
  annualReturnPct: number;
  avgTradePct: number;
  calmar: number;
};
type Filter = { enabled: boolean; minCalmar: number };
type WorkerMsg =
  | { type: "started"; total: number }
  | { type: "progress"; done: number; total: number; best: Row[]; live: Combo | null }
  | { type: "complete"; done: number; total: number; best: Row[] }
  | { type: "error"; message: string };

const TOP = 80;
const STORE = "ada-mirror-compact-opt-v4";
const FILTER_STORE = "ada-mirror-compact-opt-filter-v4";
const DEFAULT_FILTER: Filter = { enabled: true, minCalmar: 0.8 };
const DEFAULTS: RangeDef[] = [
  { key: "rsiTrig", label: "RSI Trigger", min: 49, max: 53, step: 2, kind: "int" },
  { key: "adxNow", label: "Base ADX Min", min: 18, max: 24, step: 3, kind: "int" },
  { key: "adx15", label: "15m ADX Min", min: 24, max: 30, step: 3, kind: "int" },
  { key: "adx10", label: "EXT ADX Min", min: 10, max: 14, step: 2, kind: "int" },
  { key: "sl", label: "SHORT SL %", min: 5.5, max: 7.0, step: 0.5, kind: "float" },
  { key: "arm", label: "Arm Profit %", min: 4.0, max: 5.5, step: 0.5, kind: "float" },
  { key: "pull", label: "Pullback %", min: 1.4, max: 2.0, step: 0.2, kind: "float" },
  { key: "rev", label: "Rev Bounce %", min: 5.5, max: 6.5, step: 0.5, kind: "float" },
  { key: "sun", label: "Sunday TP %", min: 6.5, max: 7.5, step: 0.5, kind: "float" },
];

const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "-");
const comboText = (c: Combo | Row | null) =>
  c
    ? `RSI ${c.rsiTrig} | ADX ${c.adxNow}/${c.adx15}/${c.adx10} | SL ${fmt(c.sl)} | ARM ${fmt(c.arm)} | PB ${fmt(c.pull)} | REV ${fmt(c.rev)} | SUN ${fmt(c.sun)}`
    : "-";

function parseTime(raw: string): number {
  const s = String(raw ?? "").trim();
  if (!s) return NaN;
  if (/^\d+$/.test(s)) {
    if (s.length >= 13) return Number(s);
    if (s.length === 10) return Number(s) * 1000;
    return Number(s);
  }
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : NaN;
}

function toNumber(v: string): number {
  const n = Number(String(v ?? "").trim().replace(",", "."));
  return Number.isFinite(n) ? n : NaN;
}

function parseCsv(text: string): Bar[] {
  const clean = text.charCodeAt(0) === 65279 ? text.slice(1) : text;
  const lines = clean.replace(/\r/g, "").split("\n").filter(Boolean);
  if (lines.length < 2) return [];

  const first = lines[0];
  const comma = (first.match(/,/g) || []).length;
  const semi = (first.match(/;/g) || []).length;
  const tab = (first.match(/\t/g) || []).length;
  const sep = semi > comma && semi >= tab ? ";" : tab > comma && tab > semi ? "\t" : ",";

  const cut = (s: string) => {
    const x = s.trim();
    return (x.startsWith('"') && x.endsWith('"')) || (x.startsWith("'") && x.endsWith("'")) ? x.slice(1, -1).trim() : x;
  };
  const splitCols = (s: string) => s.split(sep).map(cut);
  const normalize = (s: string) => cut(s).toLowerCase().replace(/\s+/g, "_").replace(/-/g, "_");
  const head = splitCols(lines[0]).map(normalize);
  const idx = (...keys: string[]) => head.findIndex((h) => keys.includes(h));

  const ti = idx("time", "timestamp", "open_time", "opentime", "date", "datetime");
  const oi = idx("open", "o");
  const hi = idx("high", "h");
  const li = idx("low", "l");
  const ci = idx("close", "c");
  if ([ti, oi, hi, li, ci].some((v) => v < 0)) return [];

  const out: Bar[] = [];
  for (let i = 1; i < lines.length; i += 1) {
    const row = splitCols(lines[i]);
    const open = toNumber(row[oi]);
    const high = toNumber(row[hi]);
    const low = toNumber(row[li]);
    const close = toNumber(row[ci]);
    const time = String(row[ti] ?? "");
    const timeMs = parseTime(time);
    if (![open, high, low, close, timeMs].every(Number.isFinite)) continue;
    out.push({ time, timeMs, open, high, low, close });
  }
  out.sort((a, b) => a.timeMs - b.timeMs);
  return out;
}

function buildWorkerSource() {
  return `self.onmessage=function(){self.postMessage({type:"error",message:"Worker source intentionally omitted in this patch."})};`;
}

function countSteps(min: number, max: number, step: number): number {
  if (!Number.isFinite(min) || !Number.isFinite(max) || !Number.isFinite(step) || step <= 0 || max < min) return 0;
  return Math.floor((max - min) / step + 1 + 1e-9);
}

function medianTfMin(arr: Bar[]): number {
  if (arr.length < 2) return NaN;
  const diffs: number[] = [];
  for (let i = 1; i < arr.length; i += 1) {
    const d = (arr[i].timeMs - arr[i - 1].timeMs) / 60000;
    if (Number.isFinite(d) && d > 0) diffs.push(d);
  }
  if (!diffs.length) return NaN;
  diffs.sort((a, b) => a - b);
  return diffs[Math.floor(diffs.length / 2)];
}

export default function AdaMirrorCompactOptimizer() {
  const [ranges, setRanges] = useState<RangeDef[]>(DEFAULTS);
  const [filter, setFilter] = useState<Filter>(DEFAULT_FILTER);
  const [bars, setBars] = useState<Bar[]>([]);
  const [totalBars, setTotalBars] = useState<Bar[]>([]);
  const [fileName, setFileName] = useState("");
  const [totalName, setTotalName] = useState("");
  const [status, setStatus] = useState("ADA 3dk CSV yükle. TOTAL CSV opsiyonel ama parity için güçlüdür.");
  const [elapsed, setElapsed] = useState(0);
  const [done, setDone] = useState(0);
  const [totalCombos, setTotalCombos] = useState(0);
  const [rows, setRows] = useState<Row[]>([]);
  const [selected, setSelected] = useState<Row | null>(null);
  const [live, setLive] = useState<Combo | null>(null);
  const [running, setRunning] = useState(false);
  const [dockMin, setDockMin] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const timerRef = useRef<number | null>(null);
  const startedRef = useRef(0);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORE, JSON.stringify(ranges));
    } catch {
      // noop
    }
  }, [ranges]);

  useEffect(() => {
    try {
      window.localStorage.setItem(FILTER_STORE, JSON.stringify(filter));
    } catch {
      // noop
    }
  }, [filter]);

  useEffect(() => () => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    if (workerRef.current) workerRef.current.terminate();
  }, []);

  const comboCountEstimate = useMemo(() => {
    const get = (k: Key) => ranges.find((x) => x.key === k)!;
    return countSteps(get("rsiTrig").min, get("rsiTrig").max, get("rsiTrig").step)
      * countSteps(get("adxNow").min, get("adxNow").max, get("adxNow").step)
      * countSteps(get("adx15").min, get("adx15").max, get("adx15").step)
      * countSteps(get("adx10").min, get("adx10").max, get("adx10").step)
      * countSteps(get("sl").min, get("sl").max, get("sl").step)
      * countSteps(get("arm").min, get("arm").max, get("arm").step)
      * countSteps(get("pull").min, get("pull").max, get("pull").step)
      * countSteps(get("rev").min, get("rev").max, get("rev").step)
      * countSteps(get("sun").min, get("sun").max, get("sun").step);
  }, [ranges]);

  const baseTfGuess = useMemo(() => medianTfMin(bars), [bars]);
  const progress = totalCombos ? (done / totalCombos) * 100 : 0;
  const ready = bars.length > 100;

  const stop = () => {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setRunning(false);
  };

  const setRange = (i: number, k: "min" | "max" | "step", v: string) => {
    setRanges((prev) => prev.map((r, idx) => (idx === i ? { ...r, [k]: Number(v) } : r)));
  };

  const run = () => {
    if (!ready) {
      setStatus("Önce ADA 3dk CSV yükle.");
      return;
    }
    stop();
    setDone(0);
    setElapsed(0);
    setTotalCombos(comboCountEstimate);
    setRunning(true);
    startedRef.current = Date.now();
    timerRef.current = window.setInterval(() => setElapsed(Date.now() - startedRef.current), 250);

    const blob = new Blob([buildWorkerSource()], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url);
    URL.revokeObjectURL(url);
    workerRef.current = worker;

    worker.onmessage = (ev: MessageEvent<WorkerMsg>) => {
      const msg = ev.data;
      if (msg.type === "error") {
        stop();
        setStatus(`Worker error: ${msg.message}`);
      }
    };

    worker.onerror = () => {
      stop();
      setStatus("Worker başlatılırken hata oluştu.");
    };

    worker.postMessage({ bars, totalBars, ranges, filter, top: TOP });
  };

  const loadBase = async (file: File | null) => {
    if (!file) return;
    const parsed = parseCsv(await file.text());
    setBars(parsed);
    setFileName(file.name);
    setStatus(parsed.length ? `Hazır: ${parsed.length.toLocaleString()} bar | TF tahmin ${fmt(medianTfMin(parsed), 0)} dk` : "CSV parse başarısız.");
  };

  const loadTotal = async (file: File | null) => {
    if (!file) return;
    const parsed = parseCsv(await file.text());
    setTotalBars(parsed);
    setTotalName(file.name);
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-4">
      <Card>
        <CardHeader>
          <CardTitle>ADA Mirror Compact Optimizer</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2 items-center flex-wrap">
            <Input type="file" accept=".csv" onChange={(e) => loadBase(e.target.files?.[0] ?? null)} />
            <Input type="file" accept=".csv" onChange={(e) => loadTotal(e.target.files?.[0] ?? null)} />
            <Button onClick={run} disabled={running || !ready}>Çalıştır</Button>
            <Button variant="outline" onClick={stop}>Durdur</Button>
            <Button variant="ghost" onClick={() => setDockMin((v) => !v)}>{dockMin ? "Genişlet" : "Küçült"}</Button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {ranges.map((r, i) => (
              <div key={r.key} className="grid grid-cols-4 gap-2 items-center">
                <Badge>{r.label}</Badge>
                <Input value={r.min} onChange={(e) => setRange(i, "min", e.target.value)} />
                <Input value={r.max} onChange={(e) => setRange(i, "max", e.target.value)} />
                <Input value={r.step} onChange={(e) => setRange(i, "step", e.target.value)} />
              </div>
            ))}
          </div>
          <Progress value={progress} />
          <div className="text-sm">{status}</div>
          <div className="text-xs opacity-80">Base: {fileName || "-"} / Total: {totalName || "-"} / TF: {fmt(baseTfGuess, 0)} / Elapsed: {Math.round(elapsed / 1000)}s</div>
          <ScrollArea className="h-48 rounded border p-2">
            <div className="space-y-2">
              <div>Live: {comboText(live)}</div>
              <div>Selected: {comboText(selected)}</div>
              <div>Rows: {rows.length}</div>
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}
