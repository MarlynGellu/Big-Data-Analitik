# =========================================================
# PREPROCESSING DATASET BSM
# METODE NAIVE BAYES - PREDIKSI KELAYAKAN PENERIMA BSM
# VERSI PYTHON / VS CODE
# =========================================================

# =========================================================
# 1. IMPORT LIBRARY
# =========================================================

import pandas as pd
import numpy as np
import os

# =========================================================
# 2. MEMBACA FILE CSV
# =========================================================
nama_file = r"C:\Users\ACER\Documents\BDA\NaiveBayes\DATA_MENTAH.csv"

raw = pd.read_csv(
    nama_file,
    sep=';',
    header=None
)

print("=" * 55)
print("DATASET BERHASIL DIBACA")
print("=" * 55)

# =========================================================
# 3. MEMBERSIHKAN DATASET
# =========================================================

# mengambil header (baris ke-2 index 2)
header = raw.iloc[2]

# mengambil isi data
df = raw[3:].copy()

# mengganti nama kolom
df.columns = header

# reset index
df.reset_index(drop=True, inplace=True)

# menghapus kolom kosong
df = df.loc[:, ~df.columns.isna()]

# merapikan nama kolom
df.columns = df.columns.str.strip()

print("HEADER BERHASIL DIRAPIKAN")
print("\nSEMUA KOLOM DATASET :")
for col in df.columns:
    print(" -", col)

# =========================================================
# 4. AMBIL HANYA KOLOM YANG DIGUNAKAN
# =========================================================

# Kolom yang dipakai untuk Naive Bayes
KOLOM_FITUR = [
    'PENDAPATAN ORANG TUA',
    'PEKERJAAN  ORANG TUA',
    'JUMLAH TANGGUNGAN',
    'STATUS RUMAH',
    'LABEL'
]

# Filter kolom yang tersedia
kolom_ada = [k for k in KOLOM_FITUR if k in df.columns]

df = df[kolom_ada].copy()

print("\nKOLOM YANG DIGUNAKAN :")
for col in df.columns:
    print(" -", col)

# =========================================================
# 5. NORMALISASI PENDAPATAN ORANG TUA
# =========================================================
# Kategori :
#   0 = Rendah   (500.000 - 1.000.000)
#   1 = Sedang   (1.000.001 - 4.000.000)
#   2 = Tinggi   (> 4.000.000)
# =========================================================

def kategori_pendapatan(x):

    x = str(x).lower()
    x = x.replace('.', '').replace(',', '')
    x = x.replace('rp', '').strip()

    angka = ''.join(filter(str.isdigit, x))

    if angka == '':
        return np.nan

    angka = int(angka)

    if 500000 <= angka <= 1000000:
        return 'Rendah'

    elif 1000001 <= angka <= 4000000:
        return 'Sedang'

    elif angka > 4000000:
        return 'Tinggi'

    else:
        return np.nan

# =========================================================
# 6. NORMALISASI PEKERJAAN ORANG TUA
# =========================================================
# Kategori :
#   Petani, PNS, Polisi, Wiraswasta, dll
# =========================================================

def kategori_pekerjaan(x):

    x = str(x).lower().strip()

    peta = {
        'petani'      : 'Petani',
        'pns'         : 'PNS',
        'polisi'      : 'Polisi',
        'wiraswasta'  : 'Wiraswasta',
        'buruh'       : 'Buruh',
        'pedagang'    : 'Pedagang',
        'nelayan'     : 'Nelayan',
        'swasta'      : 'Swasta',
        'honorer'     : 'Honorer',
        'tidak bekerja': 'Tidak Bekerja',
    }

    for kunci, nilai in peta.items():
        if kunci in x:
            return nilai

    # jika tidak ditemukan, kembalikan apa adanya (capitalize)
    if x and x != 'nan':
        return x.title()

    return np.nan

# =========================================================
# 7. NORMALISASI JUMLAH TANGGUNGAN
# =========================================================
# Kategori :
#   Sedikit  = 1 - 2
#   Sedang   = 3 - 4
#   Banyak   = >= 5
# =========================================================

def kategori_tanggungan(x):

    try:
        angka = int(float(str(x).strip()))

        if 1 <= angka <= 2:
            return 'Sedikit'

        elif 3 <= angka <= 4:
            return 'Sedang'

        elif angka >= 5:
            return 'Banyak'

        else:
            return np.nan

    except:
        return np.nan

# =========================================================
# 8. NORMALISASI STATUS RUMAH
# =========================================================
# Kategori :
#   Milik Sendiri = Layak
#   Kontrak/Sewa  = Tidak Layak
# =========================================================

def kategori_rumah(x):

    x = str(x).lower().strip()

    if 'milik' in x or 'sendiri' in x:
        return 'Milik Sendiri'

    elif 'kontrak' in x or 'sewa' in x:
        return 'Kontrak/Sewa'

    else:
        return np.nan

# =========================================================
# 9. NORMALISASI LABEL (TARGET)
# =========================================================
# Kategori :
#   Ya    = Layak menerima BSM
#   Tidak = Tidak layak
# =========================================================

def kategori_label(x):

    x = str(x).lower().strip()

    if x in ['ya', '1', 'layak', 'iya']:
        return 'Ya'

    elif x in ['tidak', '0', 'tidak layak']:
        return 'Tidak'

    else:
        return np.nan

# =========================================================
# 10. TERAPKAN NORMALISASI KE KOLOM
# =========================================================

print("\n" + "=" * 55)
print("PROSES NORMALISASI ...")
print("=" * 55)

if 'PENDAPATAN ORANG TUA' in df.columns:
    df['PENDAPATAN ORANG TUA'] = df['PENDAPATAN ORANG TUA'].apply(kategori_pendapatan)
    print("✓ PENDAPATAN ORANG TUA  → Rendah / Sedang / Tinggi")

# Nama kolom pekerjaan kadang dobel spasi
kol_pekerjaan = [c for c in df.columns if 'PEKERJAAN' in c]
if kol_pekerjaan:
    df[kol_pekerjaan[0]] = df[kol_pekerjaan[0]].apply(kategori_pekerjaan)
    df.rename(columns={kol_pekerjaan[0]: 'PEKERJAAN ORANG TUA'}, inplace=True)
    print("✓ PEKERJAAN ORANG TUA   → Petani / PNS / Wiraswasta / dll")

if 'JUMLAH TANGGUNGAN' in df.columns:
    df['JUMLAH TANGGUNGAN'] = df['JUMLAH TANGGUNGAN'].apply(kategori_tanggungan)
    print("✓ JUMLAH TANGGUNGAN     → Sedikit / Sedang / Banyak")

if 'STATUS RUMAH' in df.columns:
    df['STATUS RUMAH'] = df['STATUS RUMAH'].apply(kategori_rumah)
    print("✓ STATUS RUMAH          → Milik Sendiri / Kontrak/Sewa")

if 'LABEL' in df.columns:
    df['LABEL'] = df['LABEL'].apply(kategori_label)
    print("✓ LABEL                 → Ya / Tidak")

# =========================================================
# 11. HAPUS DATA KOSONG
# =========================================================

jumlah_sebelum = len(df)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

jumlah_sesudah = len(df)

print(f"\nData sebelum bersih : {jumlah_sebelum}")
print(f"Data setelah bersih : {jumlah_sesudah}")
print(f"Data dihapus        : {jumlah_sebelum - jumlah_sesudah}")

# =========================================================
# 12. TAMPILKAN HASIL
# =========================================================

print("\n" + "=" * 55)
print("HASIL NORMALISASI (10 data pertama)")
print("=" * 55)
print(df.head(10).to_string())

# =========================================================
# 13. DISTRIBUSI TIAP KOLOM
# =========================================================

print("\n" + "=" * 55)
print("DISTRIBUSI DATA")
print("=" * 55)

for col in df.columns:
    print(f"\n[ {col} ]")
    print(df[col].value_counts().to_string())

# =========================================================
# 14. SIMPAN HASIL
# =========================================================

df.to_csv('hasil_preprocessing_bsm.csv', index=False, sep=';')
df.to_excel('hasil_preprocessing_bsm.xlsx', index=False)

print("\n" + "=" * 55)
print("FILE BERHASIL DISIMPAN")
print("=" * 55)
print("1. hasil_preprocessing_bsm.csv")
print("2. hasil_preprocessing_bsm.xlsx")
print("=" * 55)
