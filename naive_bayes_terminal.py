# =========================================================
# NAIVE BAYES - TAMPILAN TERMINAL LENGKAP
# PREDIKSI KELAYAKAN PENERIMA BSM
# =========================================================

import pandas as pd
import numpy as np
import math
import os
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# =========================================================
# WARNA TERMINAL (ANSI)
# =========================================================

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
WHITE  = "\033[97m"
BG_BLUE= "\033[44m"

def garis(karakter="=", panjang=60, warna=CYAN):
    print(warna + karakter * panjang + RESET)

def judul(teks, warna=CYAN):
    garis("=", 60, warna)
    print(warna + BOLD + f"  {teks}" + RESET)
    garis("=", 60, warna)

def sub_judul(teks, warna=BLUE):
    print()
    print(warna + BOLD + f"--- {teks} ---" + RESET)
    garis("-", 50, warna)

# =========================================================
# 1. BACA FILE HASIL PREPROCESSING
# =========================================================

judul("SISTEM NAIVE BAYES - PREDIKSI BSM")

folder    = os.path.dirname(os.path.abspath(__file__))
file_data = os.path.join(folder, "hasil_preprocessing_bsm.csv")

if not os.path.exists(file_data):
    print(RED + "File hasil_preprocessing_bsm.csv tidak ditemukan!" + RESET)
    print(YELLOW + "Jalankan preprocessing.py terlebih dahulu." + RESET)
    exit()

df = pd.read_csv(file_data, sep=';', dtype=str)

# Pastikan kolom standar
df.columns = df.columns.str.strip()

TARGET = 'LABEL'
FITUR  = [c for c in df.columns if c != TARGET]

print(GREEN + f"\n  File       : hasil_preprocessing_bsm.csv" + RESET)
print(GREEN + f"  Total Data : {len(df)} baris" + RESET)
print(GREEN + f"  Target     : {TARGET}" + RESET)
print(GREEN + f"  Fitur      : {', '.join(FITUR)}" + RESET)

# =========================================================
# 2. TAMPILKAN DATA MENTAH (10 PERTAMA)
# =========================================================

judul("DATA SETELAH PREPROCESSING (10 PERTAMA)")

df_tampil = df.head(10).copy()
df_tampil.insert(0, "No", range(1, len(df_tampil)+1))

# Hitung lebar kolom
lebar = {col: max(len(str(col)), df_tampil[col].astype(str).str.len().max()) + 2
         for col in df_tampil.columns}

# Header tabel
header_line = BOLD + CYAN
for col in df_tampil.columns:
    header_line += f"  {str(col):<{lebar[col]}}"
header_line += RESET
print(header_line)
garis("-", 60, CYAN)

# Isi tabel
for _, row in df_tampil.iterrows():
    line = ""
    for col in df_tampil.columns:
        val = str(row[col])
        if col == TARGET:
            warna = GREEN if val == "Ya" else RED
            line += f"  {warna}{val:<{lebar[col]}}{RESET}"
        else:
            line += f"  {WHITE}{val:<{lebar[col]}}{RESET}"
    print(line)

# =========================================================
# 3. DISTRIBUSI KELAS
# =========================================================

judul("DISTRIBUSI KELAS LABEL")

total     = len(df)
count_kls = Counter(df[TARGET])

for k, v in count_kls.items():
    pct  = v / total * 100
    bar  = "█" * int(pct / 2)
    warna = GREEN if k == "Ya" else RED
    print(f"  {warna}{BOLD}{k:<10}{RESET} : {warna}{v:>4} data  ({pct:5.1f}%)  {bar}{RESET}")

# =========================================================
# 4. LANGKAH 1 — HITUNG PRIOR P(C)
# =========================================================

judul("LANGKAH 1 : HITUNG PRIOR  P(C)")

kelas_list = sorted(df[TARGET].unique())
prior      = {}

print(f"\n  {'Kelas':<15} {'Jumlah':>8} {'Total':>8} {'P(C)':>12}")
garis("-", 50, BLUE)

for k in kelas_list:
    p = count_kls[k] / total
    prior[k] = p
    print(f"  {BOLD}{k:<15}{RESET} {YELLOW}{count_kls[k]:>8}{RESET} {WHITE}{total:>8}{RESET} {GREEN}{p:>12.6f}{RESET}")

print()
for k in kelas_list:
    warna = GREEN if k == "Ya" else RED
    print(f"  {warna}P({k}) = {count_kls[k]} / {total} = {round(prior[k], 6)}{RESET}")

# =========================================================
# 5. LANGKAH 2 — HITUNG LIKELIHOOD P(xi | C)
# =========================================================

judul("LANGKAH 2 : HITUNG LIKELIHOOD  P(xi | C)")
print(YELLOW + "  Menggunakan Laplace Smoothing (alpha = 1)" + RESET)

likelihood = {}

for k in kelas_list:

    likelihood[k] = {}
    X_k = df[df[TARGET] == k]
    n_k = len(X_k)

    sub_judul(f"KELAS : {k}  (n = {n_k})", MAGENTA)

    for f in FITUR:

        nilai_unik = sorted(df[f].unique().astype(str))
        count_f    = Counter(X_k[f].astype(str))
        V          = len(nilai_unik)

        likelihood[k][f] = {}

        print(f"\n  {BOLD}{CYAN}[ {f} ]{RESET}")
        print(f"  {'Nilai':<20} {'Cacah':>6} {'Formula':<30} {'P(xi|C)':>12}")
        garis("-", 70, BLUE)

        for v in nilai_unik:
            c  = count_f.get(v, 0)
            lh = (c + 1) / (n_k + V)
            likelihood[k][f][v] = lh
            formula = f"({c}+1) / ({n_k}+{V})"
            print(f"  {WHITE}{v:<20}{RESET} {YELLOW}{c:>6}{RESET}  {formula:<30} {GREEN}{lh:>12.6f}{RESET}")

# =========================================================
# 6. CONTOH PREDIKSI MANUAL (DATA KE-1)
# =========================================================

judul("LANGKAH 3 : CONTOH PREDIKSI MANUAL (DATA KE-1)")

contoh = df.iloc[0]
print(YELLOW + "\n  Data yang diprediksi :" + RESET)
for f in FITUR:
    print(f"    {f:<25} : {WHITE}{contoh[f]}{RESET}")
print(f"    {'Label Aktual':<25} : {GREEN}{contoh[TARGET]}{RESET}")

print()
posterior_terbaik = None
kelas_terbaik     = None

for k in kelas_list:
    warna_k = GREEN if k == "Ya" else RED

    print(f"\n  {warna_k}{BOLD}=== Kelas : {k} ==={RESET}")
    log_total = math.log(prior[k])
    print(f"    log P({k}) = log({round(prior[k],6)}) = {round(log_total,6)}")

    for f in FITUR:
        val = str(contoh[f])
        lh  = likelihood[k][f].get(val, 1e-9)
        log_lh = math.log(lh + 1e-9)
        log_total += log_lh
        print(f"    log P({val} | {k}) = log({round(lh,6)}) = {round(log_lh,6)}")

    print(f"    {warna_k}{BOLD}Log Posterior = {round(log_total,6)}{RESET}")

    if posterior_terbaik is None or log_total > posterior_terbaik:
        posterior_terbaik = log_total
        kelas_terbaik     = k

print()
warna_hasil = GREEN if kelas_terbaik == "Ya" else RED
print(f"  {warna_hasil}{BOLD}>> PREDIKSI : {kelas_terbaik}{RESET}")
print(f"  {GREEN if contoh[TARGET]==kelas_terbaik else RED}>> AKTUAL   : {contoh[TARGET]}{RESET}")

# =========================================================
# 7. PREDIKSI SEMUA DATA
# =========================================================

judul("LANGKAH 4 : PREDIKSI SELURUH DATA")

y_true = []
y_pred = []

for i in range(len(df)):
    row    = df.iloc[i]
    aktual = row[TARGET]

    posterior = {}
    for k in kelas_list:
        log_p = math.log(prior[k])
        for f in FITUR:
            val  = str(row[f])
            lh   = likelihood[k][f].get(val, 1e-9)
            log_p += math.log(lh + 1e-9)
        posterior[k] = log_p

    prediksi = max(posterior, key=posterior.get)

    y_true.append(aktual)
    y_pred.append(prediksi)

# Tampilkan tabel prediksi
print(f"\n  {'No':<5} {'Pendapatan':<12} {'Pekerjaan':<14} {'Tanggungan':<12} {'Rumah':<16} {'Aktual':<10} {'Prediksi':<10} {'Status'}")
garis("-", 95, BLUE)

for i in range(len(df)):
    row     = df.iloc[i]
    aktual  = y_true[i]
    pred    = y_pred[i]
    status  = GREEN + "BENAR" + RESET if aktual == pred else RED + "SALAH" + RESET

    a_warna = GREEN if aktual == "Ya" else RED
    p_warna = GREEN if pred   == "Ya" else RED

    print(
        f"  {WHITE}{i+1:<5}{RESET}"
        f" {WHITE}{str(row.get('PENDAPATAN ORANG TUA','')):<12}{RESET}"
        f" {WHITE}{str(row.get('PEKERJAAN ORANG TUA','')):<14}{RESET}"
        f" {WHITE}{str(row.get('JUMLAH TANGGUNGAN','')):<12}{RESET}"
        f" {WHITE}{str(row.get('STATUS RUMAH','')):<16}{RESET}"
        f" {a_warna}{aktual:<10}{RESET}"
        f" {p_warna}{pred:<10}{RESET}"
        f" {status}"
    )

# =========================================================
# 8. CONFUSION MATRIX
# =========================================================

judul("LANGKAH 5 : CONFUSION MATRIX")

labels = sorted(set(y_true))
cm     = confusion_matrix(y_true, y_pred, labels=labels)

# Header
print(f"\n  {'':20}", end="")
for l in labels:
    print(f"  {CYAN}{BOLD}Pred {l:<8}{RESET}", end="")
print()
garis("-", 55, BLUE)

for i, l in enumerate(labels):
    print(f"  {YELLOW}{BOLD}Aktual {l:<13}{RESET}", end="")
    for j in range(len(labels)):
        val   = cm[i][j]
        warna = GREEN if i == j else RED
        print(f"  {warna}{BOLD}{val:>12}{RESET}", end="")
    print()

# TP TN FP FN
if len(labels) == 2:
    tn, fp, fn, tp = cm.ravel()
    print()
    print(f"  {GREEN}TP (True Positive)  = {tp:>4}  Prediksi Ya  & Aktual Ya{RESET}")
    print(f"  {GREEN}TN (True Negative)  = {tn:>4}  Prediksi Tidak & Aktual Tidak{RESET}")
    print(f"  {RED}FP (False Positive) = {fp:>4}  Prediksi Ya  tapi Aktual Tidak{RESET}")
    print(f"  {RED}FN (False Negative) = {fn:>4}  Prediksi Tidak tapi Aktual Ya{RESET}")

# =========================================================
# 9. AKURASI & CLASSIFICATION REPORT
# =========================================================

judul("LANGKAH 6 : AKURASI & CLASSIFICATION REPORT")

akurasi = accuracy_score(y_true, y_pred)

print(f"\n  {BOLD}{GREEN}AKURASI = {round(akurasi * 100, 2)} %{RESET}")
print()

# Classification report manual
report = classification_report(y_true, y_pred, digits=4)

for line in report.split('\n'):
    if any(k in line for k in kelas_list):
        print(f"  {GREEN}{line}{RESET}")
    elif 'accuracy' in line:
        print(f"  {YELLOW}{BOLD}{line}{RESET}")
    elif 'macro' in line or 'weighted' in line:
        print(f"  {CYAN}{line}{RESET}")
    else:
        print(f"  {WHITE}{line}{RESET}")

# =========================================================
# 10. RINGKASAN AKHIR
# =========================================================

judul("RINGKASAN HASIL NAIVE BAYES", GREEN)

benar = sum(1 for a, p in zip(y_true, y_pred) if a == p)
salah = len(y_true) - benar

print(f"\n  {WHITE}Total Data    : {BOLD}{len(y_true)}{RESET}")
print(f"  {GREEN}Prediksi Benar: {BOLD}{benar}{RESET}")
print(f"  {RED}Prediksi Salah: {BOLD}{salah}{RESET}")
print(f"  {YELLOW}{BOLD}Akurasi       : {round(akurasi*100,2)} %{RESET}")
print()

for k in kelas_list:
    warna = GREEN if k == "Ya" else RED
    jml   = count_kls[k]
    pct   = jml / total * 100
    print(f"  {warna}P({k}) = {round(prior[k],6)}   ({jml} dari {total} data = {pct:.1f}%){RESET}")

print()
garis("=", 60, GREEN)
print(GREEN + BOLD + "  SELESAI - Naive Bayes BSM" + RESET)
garis("=", 60, GREEN)
