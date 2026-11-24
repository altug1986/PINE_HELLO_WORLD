#!/usr/bin/env python3
"""Download XMRUSDT 5m top-trader positions from Binance Vision daily metrics.
The existing XMR METRICS canonical .csv.gz is NEVER changed by this script.
Timestamps are emitted exactly as published; do not backtest until aligned.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from pathlib import Path
from collections import Counter
import csv, gzip, io, json, time, zipfile, math

START = date(2021,12,1)
END = date(2026,11,22)
SYMBOL = "XMRUSDT"
FIELD = "sum_toptrader_long_short_ratio"
STEM = "XMR_TOP_TRADER_POSITIONS_5m_UTC_20211201_THROUGH_20261122"
OUT = Path("data")
BASE = "https://data.binance.vision/data/futures/um/daily/metrics/XMRUSDT/XMRUSDT-metrics-"

def download(day):
    url = BASE + day.isoformat() + ".zip"
    payload, error = None, ""
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent":"CYBORG-XMR-Metrics-Audit/1.0"})
            with urlopen(request, timeout=35) as res:
                payload = res.read()
            break
        except HTTPError as e:
            if e.code in (403,404):
                return day.isoformat(), "UNAVAILABLE_HTTP_" + str(e.code), [], 0
            error = "HTTP_" + str(e.code)
            if e.code not in (429,500,502,503,504):
                break
        except (URLError, OSError, TimeoutError) as e:
            error = str(e)[:100]
        time.sleep(1+attempt*2)
    if payload is None:
        return day.isoformat(), "DOWNLOAD_ERROR:" + error, [], 0
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if len(names)!=1:
                return day.isoformat(), "BAD_ZIP_FILE_COUNT", [], 0
            with z.open(names[0]) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig",newline=""))
                if not reader.fieldnames or FIELD not in reader.fieldnames or "create_time" not in reader.fieldnames:
                    return day.isoformat(), "MISSING_SOURCE_COLUMN", [], 0
                rows, total = [], 0
                for row in reader:
                    total += 1
                    if row.get("symbol", SYMBOL).strip().upper() != SYMBOL:
                        continue
                    val = (row.get(FIELD) or "").strip()
                    if not val:
                        continue
                    try:
                        n = float(val)
                        if not (math.isfinite(n) and n>0):
                            continue
                    except ValueError:
                        continue
                    raw = (row.get("create_time") or "").strip()
                    if raw:
                        rows.append((raw,val,day.isoformat()))
                if not rows:
                    return day.isoformat(), "NO_SOURCE_VALUES", [], total
                return day.isoformat(), "OK", rows, total
    except (zipfile.BadZipFile,UnicodeError,csv.Error,OSError,ValueError) as e:
        return day.isoformat(), "INVALID_ARCHIVE:" + str(e)[:80], [], 0

def main():
    OUT.mkdir(exist_ok=True)
    days=[]
    d=START
    while d<=END:
        days.append(d)
        d+=timedelta(days=1)
    results=[]
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures=[pool.submit(download,day) for day in days]
        for i,fut in enumerate(as_completed(futures),1):
            results.append(fut.result())
            if i%100==0:
                cnt=Counter(r[1] for r in results)
                print(f"Checked {i}/{len(days)}: valid={cnt['OK']} no_source={cnt['NO_SOURCE_VALUES']} 404={cnt['UNAVAILABLE_HTTP_404']}",flush=True)
    counts=Counter()
    gathered=[]
    missing=[]
    scanned=0
    for day,status,rows,total in sorted(results):
        counts[status]+=1
        scanned+=total
        if status=="OK":
            gathered.extend(rows)
        else:
            missing.append((day,status,total))
    gathered.sort(key=lambda x:(x[0],x[2]))
    distinct={}
    duplicates=0
    for ts,val,day in gathered:
        if ts in distinct:
            duplicates+=1
        else:
            distinct[ts]=(val,day)
    out_file=OUT/(STEM+".csv.gz")
    if distinct:
        with gzip.open(out_file,"wt",encoding="utf-8",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["archive_create_time_utc_RAW",FIELD,"source_file_utc_day","source"])
            for ts,(val,day) in distinct.items():
                writer.writerow([ts,val,day,"BINANCE_VISION_DAILY_METRICS"])
    with open(OUT/(STEM+"_MISSING_DAYS.csv"),"w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["archive_utc_day","status","source_rows"])
        writer.writerows(missing)
    audit={
        "symbol":SYMBOL,"interval":"5m","metric":FIELD,
        "start_utc":str(START),"end_utc":str(END),
        "days_scanned":len(days),
        "days_with_valid_values":counts["OK"],
        "days_without_valid_values":len(missing),
        "unique_valid_rows":len(distinct),
        "duplicate_raw_timestamp_rows":duplicates,
        "source_csv_rows_scanned":scanned,
        "status_counts":dict(counts),
        "source":"https://data.binance.vision/data/futures/um/daily/metrics/XMRUSDT/",
        "source_time_warning":"RAW Binance create_time. Do NOT merge or backtest before comparing timestamp semantics and verified endpoint time to the already corrected canonical XMR file.",
        "canonical_main_csv_modified":False,
        "downloaded_file":str(out_file) if distinct else None,
    }
    (OUT/(STEM+"_AUDIT.json")).write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(audit,indent=2),flush=True)
if __name__=="__main__":
    main()
