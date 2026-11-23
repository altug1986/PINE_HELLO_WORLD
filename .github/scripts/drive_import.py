#!/usr/bin/env python3
"""Drive _GELEN klasöründen inen dosyaları repoya yerleştirir.
Kurallar:
  .csv            -> data/<ad>.csv.gz (gzip)
  .csv.gz / .gz   -> data/<ad>
  .zip            -> data/<ad>
  diğerleri       -> DRIVE_GELEN/<ad>
  > 95 MB         -> .csv.gz satır bazlı parçalanır (başlık her parçada), diğerleri 90 MB'lık ikili parçalara bölünür
Kullanım: drive_import.py <indir> <repo_kökü> <işlenen_liste> <üretilen_liste>
"""
import gzip, math, os, shutil, sys

LIMIT = 95 * 1024 * 1024
CHUNK = 90 * 1024 * 1024


def split_csv_gz(path):
    size = os.path.getsize(path)
    k = math.ceil(size / CHUNK)
    with gzip.open(path, 'rt', newline='') as f:
        header = f.readline()
        n = sum(1 for _ in f)
    while True:
        per = math.ceil(n / k)
        stem = path[:-len('.csv.gz')]
        outs = [f'{stem}_part{i + 1}of{k}.csv.gz' for i in range(k)]
        with gzip.open(path, 'rt', newline='') as f:
            f.readline()
            for i, o in enumerate(outs):
                with gzip.open(o, 'wt', newline='', compresslevel=6) as w:
                    w.write(header)
                    for _ in range(per):
                        line = f.readline()
                        if not line:
                            break
                        w.write(line)
        if all(os.path.getsize(o) <= LIMIT for o in outs):
            os.remove(path)
            return outs
        for o in outs:
            os.remove(o)
        k += 1


def split_binary(path):
    size = os.path.getsize(path)
    k = math.ceil(size / CHUNK)
    outs = []
    with open(path, 'rb') as f:
        for i in range(k):
            o = f'{path}.part{i + 1:02d}of{k:02d}'
            with open(o, 'wb') as w:
                w.write(f.read(CHUNK))
            outs.append(o)
    base = os.path.basename(path)
    note = path + '.BIRLESTIR.txt'
    with open(note, 'w') as w:
        w.write(f'Birleştirme: cat {base}.part*of{k:02d} > {base}\n')
    os.remove(path)
    return outs + [note]


def main():
    indir, root, done_list, made_list = sys.argv[1:5]
    done, made = [], []
    for name in sorted(os.listdir(indir)):
        src = os.path.join(indir, name)
        if not os.path.isfile(src):
            continue
        low = name.lower()
        if low.endswith('.csv'):
            dst = os.path.join(root, 'data', name + '.gz')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, 'rb') as r, gzip.open(dst, 'wb', compresslevel=6) as w:
                shutil.copyfileobj(r, w)
        elif low.endswith('.gz') or low.endswith('.zip'):
            dst = os.path.join(root, 'data', name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        else:
            dst = os.path.join(root, 'DRIVE_GELEN', name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        outs = [dst]
        if os.path.getsize(dst) > LIMIT:
            outs = split_csv_gz(dst) if dst.lower().endswith('.csv.gz') else split_binary(dst)
        done.append(name)
        made += [os.path.relpath(o, root) for o in outs]
        print(f'{name} -> {", ".join(os.path.relpath(o, root) for o in outs)}')
    with open(done_list, 'w') as w:
        w.write(''.join(d + '\n' for d in done))
    with open(made_list, 'w') as w:
        w.write(''.join(m + '\n' for m in made))


if __name__ == '__main__':
    main()
