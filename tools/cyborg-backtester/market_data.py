"""Normalize local OHLC without substituting feeds or filling gaps."""
import csv
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo


def timestamp(value, zone):
    if value is None or value == "":
        raise ValueError("Boş zaman damgası.")
    if not isinstance(value, datetime):
        text = str(value).strip()
        if text.lstrip("-").replace(".", "", 1).isdigit() and len(text.split(".")[0]) >= 9:
            n = float(text)
            return round(n * 1000 if abs(n) < 1e11 else n if abs(n) < 1e14 else n / 1000 if abs(n) < 1e17 else n / 1e6)
        try:
            value = datetime.fromisoformat(text.replace(" UTC", "+00:00").replace("Z", "+00:00"))
        except ValueError:
            value = datetime.strptime(text, "%d.%m.%Y %H:%M")
    if value.tzinfo is None:
        origin = ZoneInfo(zone)
        first = value.replace(tzinfo=origin, fold=0)
        if first.utcoffset() != value.replace(tzinfo=origin, fold=1).utcoffset():
            raise ValueError(f"Belirsiz yerel saat {value}. UTC ofsetli bir zaman damgası kullan.")
        value = first
    return round(value.astimezone(timezone.utc).timestamp() * 1000)


def number(value):
    text = str(value).strip().replace("\u2212", "-")
    if "," in text:
        if "." in text and text.rfind(".") > text.rfind(","):
            text = text.replace(",", "")
        else:
            text = text.replace(".", "").replace(",", ".")
    result = float(text)
    if not math.isfinite(result):
        raise ValueError(f"Geçersiz sayı: {value}")
    return result


def csv_rows(f):
    sample = f.read(8192)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        header = sample.splitlines()[0] if sample else ""
        delimiter = max((",", ";", "\t"), key=header.count)
        yield from csv.reader(f, delimiter=delimiter)
        return
    yield from csv.reader(f, dialect)


@contextmanager
def input_rows(path, member=None):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]
            if member is None and len(names) == 1:
                member = names[0]
            if member not in names:
                raise ValueError(f"{path.name}: ZIP içindeki CSV adını member olarak belirt.")
            with archive.open(member) as stream, TextIOWrapper(stream, encoding="utf-8-sig", newline="") as f:
                yield csv_rows(f)
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            yield csv_rows(f)
    elif path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        book = load_workbook(path, read_only=True, data_only=True)
        try:
            yield book[member].values if member else book.active.values
        finally:
            book.close()
    else:
        raise ValueError(f"Desteklenen veri dosyaları CSV, ZIP ve XLSX. Gelen: {path.name}")


def header_key(value):
    return "".join(c for c in str(value or "").casefold() if c.isalnum())


ALIASES = {
    "time": ("opentime", "opentimestamp", "timestamp", "timestampms", "timestamputc", "datetime", "datetimeutc", "time", "date", "openutc", "timestamputc", "ts"),
    "open": ("open", "o", "açılış"),
    "high": ("high", "h", "yüksek"),
    "low": ("low", "l", "düşük"),
    "close": ("close", "c", "kapanış"),
    "volume": ("volume", "v", "basevolume", "vol", "hacim", "volumebasexmr"),
}


def minutes(tf):
    key = str(tf).upper()
    if key in ("D", "1D"):
        return 1440
    if not key.isascii() or not key.isdecimal() or not 1 <= int(key) <= 1440:
        raise ValueError(f"24/7 dakika grafiği veya D bekleniyor: {tf}")
    return int(key)


def normalize_feed(config, base, target, start, end, allow_gaps):
    path = (base / config["path"]).resolve(strict=True)
    step = minutes(config["timeframe"]) * 60_000
    zone = config.get("timezone", "UTC")
    start = start if start is not None else -math.inf
    end = end if end is not None else math.inf
    first = last = None
    count = gaps = 0
    with input_rows(path, config.get("member")) as rows, target.open("w", newline="") as out:
        raw_header = next(rows, ())
        headers = [header_key(x) for x in raw_header]
        indices = {}
        for key, aliases in ALIASES.items():
            matches = [headers.index(name) for name in aliases if name in headers]
            if matches:
                indices[key] = matches[0]
        if len(indices) < 5:
            if raw_header and str(raw_header[0]).strip().isdigit() and len(raw_header) >= 6:
                from itertools import chain

                rows = chain([raw_header], rows)
                indices = dict(zip(ALIASES, range(6)))
            else:
                raise ValueError(f"{path.name}: timestamp, open, high, low, close kolonları gerekli. Başlık: {raw_header}")
        writer = csv.writer(out, lineterminator="\n")
        previous = None
        for row in rows:
            if not row or not any(v is not None and v != "" for v in row):
                continue
            t = timestamp(row[indices["time"]], zone)
            if previous is not None and t <= previous:
                raise ValueError(f"{path.name}: tekrarlı veya sırasız mum: {t}")
            previous = t
            if t < start or t > end:
                continue
            if t % step:
                raise ValueError(f"{path.name}: mum UTC dakika ızgarasına hizalı değil: {t}.")
            o, h, l, c = (number(row[indices[k]]) for k in ("open", "high", "low", "close"))
            v = number(row[indices["volume"]]) if "volume" in indices else 0
            if h < max(o, l, c) or l > min(o, h, c) or v < 0:
                raise ValueError(f"{path.name}: OHLC sırası veya hacim geçersiz: {t}")
            if last is not None:
                gaps += (t - last) // step - 1
            first = t if first is None else first
            last = t
            count += 1
            writer.writerow([t, t + step, o, h, l, c, v])
    if not count:
        raise ValueError(f"{path.name}: seçilen aralıkta OHLC yok.")
    if gaps and not allow_gaps:
        raise ValueError(f"{path.name}: seçilen veri aralığında {gaps:,} eksik mum var.")
    return {"bars": count, "first_open_ms": first, "last_open_ms": last, "missing_bars": gaps, "has_volume": "volume" in indices}
