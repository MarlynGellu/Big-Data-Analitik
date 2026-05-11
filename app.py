# =========================================================
# APLIKASI NAIVE BAYES - KELAYAKAN PENERIMA BSM
# STREAMLIT VERSI LENGKAP
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import os
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import plotly.graph_objects as go
import plotly.figure_factory as ff

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title  = "Naive Bayes BSM",
    page_icon   = "🎀",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# =========================================================
# TEMA WARNA
# =========================================================

TEMA = "pink"   # ganti "biru" untuk tema biru

if TEMA == "pink":
    C1 = "#fdf2f8"; C2 = "#fce7f3"; C3 = "#f9a8d4"
    C4 = "#ec4899"; C5 = "#ffffff"; C6 = "#831843"
    C8 = "#db2777"; ICON = "🎀"
else:
    C1 = "#eff6ff"; C2 = "#dbeafe"; C3 = "#93c5fd"
    C4 = "#2563eb"; C5 = "#ffffff"; C6 = "#1e3a5f"
    C8 = "#1d4ed8"; ICON = "🎓"

# =========================================================
# CSS
# =========================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
html,body,[class*="css"]{{font-family:'Nunito',sans-serif;color:{C6};}}
.stApp{{background-color:{C1};}}
section[data-testid="stSidebar"]{{background:linear-gradient(180deg,{C2} 0%,{C5} 100%);border-right:2px solid {C3}40;}}
.stButton>button{{background:linear-gradient(135deg,{C3},{C4})!important;color:{C5}!important;border:none!important;border-radius:14px!important;font-family:'Nunito',sans-serif!important;font-weight:800!important;font-size:.95rem!important;padding:.55rem 1.4rem!important;box-shadow:0 4px 14px {C3}60!important;transition:all .25s ease!important;}}
.stButton>button:hover{{transform:translateY(-2px)!important;box-shadow:0 8px 22px {C4}50!important;}}
.stSelectbox>div>div{{background:{C5}!important;border:2px solid {C3}80!important;border-radius:12px!important;color:{C6}!important;font-weight:600!important;}}
.stSelectbox label{{color:{C6}!important;font-weight:700!important;}}
[data-testid="stFileUploader"]{{background:{C5}!important;border:2px dashed {C3}!important;border-radius:16px!important;padding:1rem!important;}}
.stTabs [data-baseweb="tab-list"]{{background:{C5};border-radius:14px;padding:6px;gap:6px;box-shadow:0 2px 12px {C3}30;}}
.stTabs [data-baseweb="tab"]{{border-radius:10px!important;color:{C6}!important;font-weight:700!important;font-size:.9rem!important;padding:.4rem 1rem!important;}}
.stTabs [aria-selected="true"]{{background:linear-gradient(135deg,{C3},{C4})!important;color:{C5}!important;box-shadow:0 3px 10px {C3}60!important;}}
.stDataFrame{{border-radius:14px!important;overflow:hidden!important;border:1px solid {C3}40!important;}}
.stSuccess{{background:#f0fdf4!important;border-left:4px solid #22c55e!important;border-radius:10px!important;}}
.stError{{background:#fff1f2!important;border-left:4px solid #f43f5e!important;border-radius:10px!important;}}
.stWarning{{background:#fffbeb!important;border-left:4px solid #f59e0b!important;border-radius:10px!important;}}
.stInfo{{background:{C2}!important;border-left:4px solid {C3}!important;border-radius:10px!important;}}
.stDownloadButton>button{{background:{C5}!important;color:{C4}!important;border:2px solid {C3}!important;border-radius:14px!important;font-weight:800!important;transition:all .2s!important;}}
.stDownloadButton>button:hover{{background:{C2}!important;transform:translateY(-1px)!important;}}
.stRadio label{{background:{C5};border:1.5px solid {C3}60;border-radius:10px;padding:.4rem .8rem;cursor:pointer;color:{C6}!important;font-weight:600!important;transition:all .2s;}}

.header-box{{background:linear-gradient(135deg,{C5} 0%,{C2} 100%);border:2px solid {C3}60;border-radius:22px;padding:2rem 2.5rem;margin-bottom:1.8rem;text-align:center;box-shadow:0 6px 30px {C3}25;}}
.header-box h1{{font-size:2.2rem;font-weight:900;color:{C4};margin:0;}}
.header-box p{{color:{C6}99;font-size:1rem;margin-top:.4rem;font-weight:600;}}
.card{{background:{C5};border:1.5px solid {C3}50;border-radius:18px;padding:1.4rem 1.6rem;margin-bottom:1.2rem;box-shadow:0 3px 16px {C3}15;}}
.card-title{{font-size:1rem;font-weight:800;color:{C4};border-bottom:2px solid {C3}40;padding-bottom:.5rem;margin-bottom:.9rem;}}
.metric-pill{{background:linear-gradient(135deg,{C2},{C5});border:2px solid {C3}60;border-radius:16px;padding:1rem .8rem;text-align:center;box-shadow:0 3px 12px {C3}20;}}
.metric-pill .val{{font-size:2rem;font-weight:900;color:{C4};line-height:1;}}
.metric-pill .lbl{{font-size:.8rem;font-weight:700;color:{C6}80;margin-top:.3rem;}}
.badge-ya{{display:inline-block;background:linear-gradient(135deg,#22c55e,#16a34a);color:white;font-size:1.3rem;font-weight:900;padding:.7rem 2.2rem;border-radius:50px;box-shadow:0 6px 20px #22c55e50;}}
.badge-tidak{{display:inline-block;background:linear-gradient(135deg,#f43f5e,#dc2626);color:white;font-size:1.3rem;font-weight:900;padding:.7rem 2.2rem;border-radius:50px;box-shadow:0 6px 20px #f43f5e50;}}
.step-badge{{display:inline-block;background:linear-gradient(135deg,{C3},{C4});color:white;font-size:.8rem;font-weight:800;padding:.25rem .7rem;border-radius:20px;margin-right:.5rem;}}
.info-box{{background:linear-gradient(135deg,{C2},{C5});border-left:4px solid {C4};border-radius:12px;padding:1rem 1.2rem;margin-bottom:1rem;}}
.info-box h4{{color:{C4};margin:0 0 .4rem;font-size:1rem;}}
.info-box p{{color:{C6};margin:0;font-size:.88rem;line-height:1.7;}}
.keterangan-box{{background:linear-gradient(135deg,{C2},{C5});border:1.5px solid {C3}60;border-radius:14px;padding:.85rem 1.1rem;margin-bottom:1rem;}}
.keterangan-box .ket-title{{font-size:.82rem;font-weight:900;color:{C4};margin-bottom:.5rem;}}
.keterangan-box .ket-item{{font-size:.8rem;color:{C6};line-height:2;font-weight:600;}}
footer{{visibility:hidden;}}
#MainMenu{{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

for key, val in {
    'dataset'     : pd.DataFrame(),
    'df_tampil'   : pd.DataFrame(),
    'df_train'    : pd.DataFrame(),
    'df_test'     : pd.DataFrame(),
    'nb_model'    : None,
    'data_uji'    : pd.DataFrame(),
    'hasil_uji'   : pd.DataFrame(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================================================
# GAUSSIAN NAIVE BAYES WRAPPER
# =========================================================

def gaussian_prob(x, mean, std):
    """Hitung probabilitas Gaussian P(x|C) secara manual."""
    if std == 0 or pd.isna(std):
        std = 1e-4
    exp = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exp

class GaussianNBManual:
    def __init__(self):
        self.model  = GaussianNB()
        self.prior  = {}
        self.mean_  = {}
        self.std_   = {}
        self.kelas  = []
        self.fitur  = []

    def fit(self, X, y):
        self.fitur = list(X.columns)
        self.kelas = sorted(y.unique().tolist())
        self.model.fit(X, y)
        for k in self.kelas:
            Xk = X[y == k]
            self.prior[k]  = len(Xk) / len(y)
            self.mean_[k]  = Xk.mean()
            self.std_[k]   = Xk.std().replace(0, 1e-4)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return self.model.predict(X).tolist()
        return self.model.predict(X).tolist()

    def predict_one(self, row_dict):
        row_df = pd.DataFrame([row_dict])[self.fitur]
        return self.model.predict(row_df)[0]

    def predict_proba_log(self, row_dict):
        h = {}
        for k in self.kelas:
            lp = np.log(self.prior[k])
            for f in self.fitur:
                p = gaussian_prob(row_dict[f], self.mean_[k][f], self.std_[k][f])
                lp += np.log(p + 1e-300)
            h[k] = lp
        return h

# =========================================================
# PREPROCESSING
# =========================================================

def _norm_pendapatan(x):
    x = str(x).lower().replace('.', '').replace(',', '').replace('rp', '').strip()
    angka = ''.join(filter(str.isdigit, x))
    if not angka:
        return np.nan
    angka = int(angka)
    if 500_000 <= angka <= 1_000_000:
        return 0
    if 1_000_001 <= angka <= 4_000_000:
        return 1
    if angka > 4_000_000:
        return 2
    return np.nan

def _norm_pekerjaan(x):
    x = str(x).lower().strip()
    peta = {
        'petani'       : 0,
        'pns'          : 1,
        'polisi'       : 2,
        'wiraswasta'   : 3,
        'buruh'        : 4,
        'pedagang'     : 5,
        'nelayan'      : 6,
        'swasta'       : 7,
        'honorer'      : 8,
        'tidak bekerja': 9,
    }
    for k, v in peta.items():
        if k in x:
            return v
    return np.nan

def _norm_tanggungan(x):
    try:
        n = int(float(str(x).strip()))
        if n >= 1:
            return n
        return np.nan
    except:
        return np.nan

def _norm_rumah(x):
    x = str(x).lower().strip()
    if 'milik' in x or 'sendiri' in x:
        return 1
    if 'kontrak' in x or 'sewa' in x:
        return 0
    return np.nan

def _norm_label(x):
    x = str(x).lower().strip()
    if x in ['ya', '1', 'layak', 'iya']:
        return 1
    if x in ['tidak', '0', 'tidak layak']:
        return 0
    return np.nan

def _cari_kolom(df, kata):
    for c in df.columns:
        if kata.upper() in str(c).upper():
            return c
    return None

def _deteksi_header(df_raw):
    KATA = ['PENDAPATAN', 'PEKERJAAN', 'TANGGUNGAN', 'STATUS', 'LABEL']
    if len(df_raw) > 3:
        baris2 = df_raw.iloc[2].astype(str).str.upper()
        gabung2 = ' '.join(baris2.values)
        if sum(1 for k in KATA if k in gabung2) >= 3:
            df_raw.columns = df_raw.iloc[2].astype(str).str.strip()
            df_raw = df_raw.iloc[3:].copy()
            df_raw.reset_index(drop=True, inplace=True)
            return df_raw
    for i in range(min(10, len(df_raw))):
        baris = df_raw.iloc[i].astype(str).str.upper()
        gabung = ' '.join(baris.values)
        if sum(1 for k in KATA if k in gabung) >= 3:
            df_raw.columns = df_raw.iloc[i].astype(str).str.strip()
            df_raw = df_raw.iloc[i+1:].copy()
            df_raw.reset_index(drop=True, inplace=True)
            return df_raw
    df_raw.columns = df_raw.columns.astype(str).str.strip()
    return df_raw

def auto_preprocess(df_raw):
    df_raw = df_raw.fillna('').astype(str)
    df = _deteksi_header(df_raw)
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.isin(['nan', 'None', '', 'NaN'])]
    df = df.loc[:, df.columns.notna()]
    df = df.loc[~(df == '').all(axis=1)]
    df.reset_index(drop=True, inplace=True)
    df.columns = (
        df.columns.str.upper()
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    cp = _cari_kolom(df, 'PENDAPATAN')
    ck = _cari_kolom(df, 'PEKERJAAN')
    ct = _cari_kolom(df, 'TANGGUNGAN')
    cr = _cari_kolom(df, 'STATUS RUMAH') or _cari_kolom(df, 'RUMAH')
    cl = _cari_kolom(df, 'LABEL')
    df_tampil = df.copy()
    if cp: df_tampil[cp] = df[cp].apply(_norm_pendapatan)
    if ck: df_tampil[ck] = df[ck].apply(_norm_pekerjaan)
    if ct: df_tampil[ct] = df[ct].apply(_norm_tanggungan)
    if cr: df_tampil[cr] = df[cr].apply(_norm_rumah)
    if cl: df_tampil[cl] = df[cl].apply(_norm_label)
    peta = {}
    if cp: peta['PENDAPATAN ORANG TUA'] = df[cp].apply(_norm_pendapatan)
    if ck: peta['PEKERJAAN ORANG TUA']  = df[ck].apply(_norm_pekerjaan)
    if ct: peta['JUMLAH TANGGUNGAN']    = df[ct].apply(_norm_tanggungan)
    if cr: peta['STATUS RUMAH']         = df[cr].apply(_norm_rumah)
    if cl: peta['LABEL']                = df[cl].apply(_norm_label)
    df_model = pd.DataFrame(peta)
    valid_idx = df_model.dropna().index
    df_model  = df_model.loc[valid_idx].reset_index(drop=True)
    df_tampil = df_tampil.loc[valid_idx].reset_index(drop=True)
    return df_tampil, df_model

def latih_model(df):
    X = df.drop(columns=['LABEL'])
    y = df['LABEL']
    m = GaussianNBManual()
    m.fit(X, y)
    return m

def baca_file(file):
    if file.name.endswith(".csv"):
        for sep in [';', ',', '\t']:
            try:
                file.seek(0)
                df = pd.read_csv(file, sep=sep, header=0, dtype=str)
                if len(df.columns) > 1:
                    return df
            except:
                continue
        file.seek(0)
        return pd.read_csv(file, dtype=str)
    else:
        df = pd.read_excel(file, header=None, dtype=str)
        df = df.fillna('').astype(str)
        return df

def df_to_excel_bytes(df):
    buf=io.BytesIO(); df.to_excel(buf,index=False); return buf.getvalue()

def df_to_csv_bytes(df):
    return df.to_csv(index=False,sep=';').encode('utf-8')

# =========================================================
# PLOTLY THEME
# =========================================================

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(color=C6, family="Nunito"),
    margin       =dict(l=20,r=20,t=55,b=20),
)

# =========================================================
# LABEL DESKRIPSI (untuk tampilan di selectbox)
# =========================================================

# Keterangan kode numerik untuk ditampilkan di UI
KETERANGAN_PENDAPATAN = {
    0: "0 = Rendah (Rp 500rb – Rp 1jt)",
    1: "1 = Sedang (Rp 1jt – Rp 4jt)",
    2: "2 = Tinggi (> Rp 4jt)",
}
KETERANGAN_PEKERJAAN = {
    0: "0 = Petani",
    1: "1 = PNS (Pegawai Negeri Sipil)",
    2: "2 = Polisi",
    3: "3 = Wiraswasta",
    4: "4 = Buruh",
    5: "5 = Pedagang",
    6: "6 = Nelayan",
    7: "7 = Swasta",
    8: "8 = Honorer",
    9: "9 = Tidak Bekerja",
}
KETERANGAN_RUMAH = {
    0: "0 = Kontrak / Sewa",
    1: "1 = Milik Sendiri",
}
KETERANGAN_LABEL = {
    0: "0 = Tidak Layak BSM",
    1: "1 = Layak BSM (Ya)",
}

def render_keterangan(judul, keterangan_dict):
    """Render kotak keterangan kode numerik."""
    items_html = "".join([
        f"<div class='ket-item'><b style='color:{C4};'>{k}</b> → {v}</div>"
        for k, v in keterangan_dict.items()
    ])
    st.markdown(f"""
    <div class='keterangan-box'>
        <div class='ket-title'>📌 {judul}</div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:1.2rem 0 .8rem;'>
        <span style='font-size:2.8rem;'>{ICON}</span>
        <div style='font-size:1.2rem;font-weight:900;color:{C4};margin-top:.3rem;'>Naive Bayes BSM</div>
        <div style='font-size:.8rem;color:{C6}80;font-weight:600;'>Bantuan Siswa Miskin</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    menu = st.radio("Navigasi Menu", [
        "🏠 Dashboard",
        "📂 Upload & Training",
        "🔬 Preprocessing",
        "🧮 Perhitungan Manual",
        "📊 Evaluasi & Grafik",
        "🔍 Prediksi Data Baru",
        "📝 Data Uji",
        "💾 Unduh Hasil",
        "ℹ️ Tentang Sistem",
    ], label_visibility="collapsed")

    st.markdown("---")
    if not st.session_state.dataset.empty:
        df_s=st.session_state.dataset
        vc_s=df_s['LABEL'].value_counts() if 'LABEL' in df_s.columns else {}
        st.markdown(f"<div class='card' style='padding:.9rem;'><div class='card-title'>📋 Dataset</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:.88rem;color:{C6};'><b>Total :</b> {len(df_s)} data</div>", unsafe_allow_html=True)
        for k,v in vc_s.items():
            lbl_k = "Ya (Layak)" if k==1 else ("Tidak Layak" if k==0 else str(k))
            w="#22c55e" if k==1 else "#f43f5e"
            st.markdown(f"<div style='color:{w};font-weight:700;font-size:.85rem;'>● {lbl_k} : {v}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:.72rem;color:{C6}50;text-align:center;margin-top:.5rem;font-weight:600;'>© 2025 · Naive Bayes BSM</div>", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(f"""
<div class='header-box'>
    <h1>{ICON} Sistem Prediksi Penerima BSM</h1>
    <p>Metode Naive Bayes · Bantuan Siswa Miskin · Prediksi Kelayakan Siswa</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 🏠 DASHBOARD
# =========================================================

if menu == "🏠 Dashboard":

    st.markdown("## 🏠 Dashboard Sistem")

    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>📖 Tentang Sistem</div>
            <div style='font-size:.92rem;color:{C6};line-height:2;'>
            Sistem ini merupakan aplikasi prediksi kelayakan penerima
            <b>Bantuan Siswa Miskin (BSM)</b> menggunakan algoritma
            <b>Naive Bayes</b>.<br><br>
            Naive Bayes adalah algoritma klasifikasi berbasis
            <b>Teorema Bayes</b> yang mengasumsikan setiap fitur
            bersifat <i>independen</i> satu sama lain.
            Algoritma ini terkenal karena sederhana, cepat,
            dan memiliki performa baik untuk data kategorikal.<br><br>
            <b>Rumus utama :</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"P(C \mid X) = \frac{P(C) \times \prod_{i=1}^{n} P(x_i \mid C)}{P(X)}")
        st.markdown(f"""
        <div style='font-size:.88rem;color:{C6}99;padding:.5rem 0;'>
        Karena P(X) konstan untuk semua kelas, maka klasifikasi dilakukan dengan memilih kelas
        yang memaksimalkan <b>P(C) × ∏ P(xᵢ | C)</b>.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>🎯 Variabel yang Digunakan</div>
            <div style='font-size:.88rem;color:{C6};line-height:2.2;'>
            <span style='color:{C4};font-weight:800;'>① Pendapatan Orang Tua</span><br>
            &nbsp;&nbsp;0=Rendah / 1=Sedang / 2=Tinggi<br>
            <span style='color:{C4};font-weight:800;'>② Pekerjaan Orang Tua</span><br>
            &nbsp;&nbsp;0=Petani / 1=PNS / 2=Polisi / dst.<br>
            <span style='color:{C4};font-weight:800;'>③ Jumlah Tanggungan</span><br>
            &nbsp;&nbsp;Angka asli (1, 2, 3, …)<br>
            <span style='color:{C4};font-weight:800;'>④ Status Rumah</span><br>
            &nbsp;&nbsp;0=Kontrak/Sewa / 1=Milik Sendiri<br>
            <span style='color:#22c55e;font-weight:800;'>⑤ Label (Target)</span><br>
            &nbsp;&nbsp;1=Ya (Layak) / 0=Tidak (Tidak Layak)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🗺️ Alur Penggunaan Sistem")

    steps = [
        ("1", "📂 Upload & Training",  "Upload file dataset CSV/Excel. Sistem otomatis melakukan preprocessing dan melatih model Naive Bayes."),
        ("2", "🔬 Preprocessing",      "Lihat hasil normalisasi data — pendapatan, pekerjaan, tanggungan, dan status rumah dikonversi ke kategori."),
        ("3", "🧮 Perhitungan Manual", "Lihat detail perhitungan Prior P(C) dan Likelihood P(xi|C) secara manual lengkap dengan formula."),
        ("4", "📊 Evaluasi & Grafik",  "Lihat Confusion Matrix, grafik akurasi, precision, recall, dan F1-Score hasil model."),
        ("5", "🔍 Prediksi Data Baru", "Masukkan data siswa baru dan dapatkan prediksi kelayakan BSM secara langsung."),
        ("6", "📝 Data Uji",           "Kelola data uji — tambah manual atau import file, lalu prediksi semua sekaligus."),
        ("7", "💾 Unduh Hasil",        "Download semua hasil : dataset bersih, hasil prediksi, evaluasi model dalam format CSV/Excel."),
    ]

    col_a, col_b = st.columns(2)
    for i, (no, judul_s, deskripsi) in enumerate(steps):
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(f"""
            <div class='card' style='padding:1rem 1.2rem;'>
                <div style='display:flex;align-items:flex-start;gap:.8rem;'>
                    <div style='background:linear-gradient(135deg,{C3},{C4});color:white;
                         font-weight:900;font-size:1.1rem;min-width:36px;height:36px;
                         border-radius:50%;display:flex;align-items:center;
                         justify-content:center;flex-shrink:0;'>{no}</div>
                    <div>
                        <div style='font-weight:800;color:{C4};font-size:.95rem;'>{judul_s}</div>
                        <div style='font-size:.85rem;color:{C6};margin-top:.3rem;line-height:1.6;'>{deskripsi}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    if not st.session_state.dataset.empty:
        df=st.session_state.dataset
        st.markdown("### 📊 Statistik Dataset Aktif")
        c1,c2,c3,c4 = st.columns(4)
        total=len(df); vc=df['LABEL'].value_counts()
        ya=vc.get(1,0); tidak=vc.get(0,0)
        akurasi_info = ""
        if st.session_state.nb_model:
            model=st.session_state.nb_model
            yp=model.predict(df.drop(columns=['LABEL']))
            akurasi_info = f"{round(accuracy_score(df['LABEL'].tolist(), yp)*100,2)}%"

        for col, val, lbl_s, warna in [
            (c1, total, "Total Data", C4),
            (c2, ya, "Layak (Ya)", "#22c55e"),
            (c3, tidak, "Tidak Layak", "#f43f5e"),
            (c4, akurasi_info or "-", "Akurasi Model", C4),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-pill'>
                    <div class='val' style='color:{warna};'>{val}</div>
                    <div class='lbl'>{lbl_s}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("💡 Belum ada dataset. Silakan ke menu **📂 Upload & Training** untuk memulai.")

# =========================================================
# 📂 UPLOAD & TRAINING
# =========================================================

elif menu == "📂 Upload & Training":

    st.markdown("## 📂 Upload Dataset & Training Model")

    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>📌 Petunjuk Upload</div>
            <div style='font-size:.9rem;line-height:1.9;color:{C6};'>
            Unggah file <b>CSV</b> (pemisah <b>;</b>) atau <b>Excel (.xlsx)</b><br>
            Kolom wajib yang harus ada di file :<br>
            <span style='color:{C4};'>●</span> PENDAPATAN ORANG TUA &nbsp;
            <span style='color:{C4};'>●</span> PEKERJAAN ORANG TUA<br>
            <span style='color:{C4};'>●</span> JUMLAH TANGGUNGAN &nbsp;&nbsp;&nbsp;
            <span style='color:{C4};'>●</span> STATUS RUMAH<br>
            <span style='color:{C4};'>●</span> LABEL (isi: <b>Ya</b> atau <b>Tidak</b>)<br><br>
            Kolom lain (NAMA, NIK, NISN, dll) otomatis diabaikan.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>⚙️ Proses Otomatis</div>
            <div style='font-size:.88rem;color:{C6};line-height:2;'>
            ✅ Deteksi header otomatis<br>
            ✅ Normalisasi pendapatan<br>
            ✅ Normalisasi pekerjaan<br>
            ✅ Normalisasi tanggungan<br>
            ✅ Normalisasi status rumah<br>
            ✅ Split 80% Latih / 20% Uji<br>
            ✅ Training model Naive Bayes
            </div>
        </div>
        """, unsafe_allow_html=True)

    file = st.file_uploader("Pilih File Dataset Training", type=["csv","xlsx"])

    if file:
        try:
            df_raw = baca_file(file)
            st.success(f"✅ File **{file.name}** berhasil dibaca — {len(df_raw)} baris")

            with st.expander("👁 Lihat Data Mentah (10 baris pertama)"):
                st.dataframe(df_raw.head(10), width="stretch")

            with st.spinner("⏳ Preprocessing & Training model..."):
                df_tampil, df_model = auto_preprocess(df_raw.copy())

            if df_model.empty:
                st.error("❌ Kolom tidak ditemukan. Pastikan format file sesuai.")
            else:
                df_train, df_test = train_test_split(
                    df_model, test_size=0.2, random_state=42, stratify=df_model['LABEL']
                )
                df_train = df_train.reset_index(drop=True)
                df_test  = df_test.reset_index(drop=True)

                st.session_state.df_tampil = df_tampil
                st.session_state.dataset   = df_model
                st.session_state.df_train  = df_train
                st.session_state.df_test   = df_test
                st.session_state.nb_model  = latih_model(df_train)

                model = st.session_state.nb_model
                st.success(f"✅ Selesai! **{len(df_model)} data** valid diproses → **{len(df_train)} latih** & **{len(df_test)} uji** · Model terlatih.")

                st.markdown(f"""
                <div style='background:linear-gradient(135deg,{C2},{C5});border:2px solid {C3}60;
                    border-radius:18px;padding:1.2rem 1.5rem;margin:1rem 0;
                    box-shadow:0 4px 16px {C3}20;'>
                    <div style='font-size:.95rem;font-weight:900;color:{C4};margin-bottom:.8rem;'>
                        📐 Pembagian Dataset (Stratified Split)
                    </div>
                    <div style='display:flex;gap:1rem;flex-wrap:wrap;'>
                        <div style='flex:1;background:{C5};border:1.5px solid {C3}50;border-radius:12px;
                            padding:.8rem 1rem;text-align:center;min-width:120px;'>
                            <div style='font-size:1.6rem;font-weight:900;color:{C4};'>{len(df_model)}</div>
                            <div style='font-size:.78rem;font-weight:700;color:{C6}80;'>Total Dataset</div>
                        </div>
                        <div style='flex:1;background:{C5};border:2px solid #22c55e;border-radius:12px;
                            padding:.8rem 1rem;text-align:center;min-width:120px;'>
                            <div style='font-size:1.6rem;font-weight:900;color:#22c55e;'>{len(df_train)}</div>
                            <div style='font-size:.78rem;font-weight:700;color:{C6}80;'>Data Latih (80%)</div>
                        </div>
                        <div style='flex:1;background:{C5};border:2px solid #f59e0b;border-radius:12px;
                            padding:.8rem 1rem;text-align:center;min-width:120px;'>
                            <div style='font-size:1.6rem;font-weight:900;color:#f59e0b;'>{len(df_test)}</div>
                            <div style='font-size:.78rem;font-weight:700;color:{C6}80;'>Data Uji (20%)</div>
                        </div>
                    </div>
                    <div style='margin-top:.9rem;background:{C2};border-radius:8px;height:16px;overflow:hidden;'>
                        <div style='width:80%;height:100%;background:linear-gradient(90deg,#22c55e,#16a34a);
                            border-radius:8px;display:inline-block;'></div>
                        <div style='width:20%;height:100%;background:linear-gradient(90deg,#f59e0b,#d97706);
                            border-radius:0 8px 8px 0;display:inline-block;'></div>
                    </div>
                    <div style='display:flex;justify-content:space-between;margin-top:.3rem;'>
                        <span style='font-size:.75rem;font-weight:700;color:#22c55e;'>80% — Data Latih</span>
                        <span style='font-size:.75rem;font-weight:700;color:#f59e0b;'>20% — Data Uji</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📊 Statistik Data Latih (80%)")
                vc_train = df_train['LABEL'].value_counts()
                vc_test  = df_test['LABEL'].value_counts()
                yp_train = model.predict(df_train.drop(columns=['LABEL']))
                acc_train = accuracy_score(df_train['LABEL'].tolist(), yp_train)
                yp_test  = model.predict(df_test.drop(columns=['LABEL']))
                acc_test  = accuracy_score(df_test['LABEL'].tolist(), yp_test)

                c1,c2,c3,c4 = st.columns(4)
                for col, val, lbl_s, warna in [
                    (c1, len(df_train),        "Total Data Latih",   C4),
                    (c2, vc_train.get(1,0),    "Label Ya (latih)",   "#22c55e"),
                    (c3, vc_train.get(0,0),    "Label Tidak (latih)","#f43f5e"),
                    (c4, f"{round(acc_train*100,2)}%","Akurasi Data Latih", C4),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class='metric-pill'>
                            <div class='val' style='color:{warna};'>{val}</div>
                            <div class='lbl'>{lbl_s}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Statistik Data Uji (20%)")
                c1,c2,c3,c4 = st.columns(4)
                for col, val, lbl_s, warna in [
                    (c1, len(df_test),        "Total Data Uji",     "#f59e0b"),
                    (c2, vc_test.get(1,0),    "Label Ya (uji)",     "#22c55e"),
                    (c3, vc_test.get(0,0),    "Label Tidak (uji)",  "#f43f5e"),
                    (c4, f"{round(acc_test*100,2)}%","Akurasi Data Uji","#f59e0b"),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class='metric-pill'>
                            <div class='val' style='color:{warna};'>{val}</div>
                            <div class='lbl'>{lbl_s}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                vc = df_model['LABEL'].value_counts()
                col_pie1, col_pie2 = st.columns(2)
                with col_pie1:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(vc_train.index), values=list(vc_train.values),
                        hole=.5, marker_colors=["#22c55e","#f43f5e"],
                        textinfo="label+percent", textfont_size=13,
                    )])
                    fig_pie.update_layout(
                        title=dict(text="Distribusi Label — Data Latih (80%)", font=dict(size=13)),
                        height=260, showlegend=False, **PLOT_LAYOUT)
                    st.plotly_chart(fig_pie, width="stretch", key="pie_train")
                with col_pie2:
                    fig_pie2 = go.Figure(data=[go.Pie(
                        labels=list(vc_test.index), values=list(vc_test.values),
                        hole=.5, marker_colors=["#22c55e","#f43f5e"],
                        textinfo="label+percent", textfont_size=13,
                    )])
                    fig_pie2.update_layout(
                        title=dict(text="Distribusi Label — Data Uji (20%)", font=dict(size=13)),
                        height=260, showlegend=False, **PLOT_LAYOUT)
                    st.plotly_chart(fig_pie2, width="stretch", key="pie_test")

                tab_l, tab_u = st.tabs(["📋 Tabel Data Latih (80%)", "📋 Tabel Data Uji (20%)"])
                with tab_l:
                    df_prev = df_train.copy(); df_prev.insert(0,"No",range(1,len(df_prev)+1))
                    st.dataframe(df_prev, width="stretch", height=350)
                with tab_u:
                    df_prev2 = df_test.copy(); df_prev2.insert(0,"No",range(1,len(df_prev2)+1))
                    st.dataframe(df_prev2, width="stretch", height=350)

        except Exception as e:
            st.error(f"❌ Error : {e}")

# =========================================================
# 🔬 PREPROCESSING
# =========================================================

elif menu == "🔬 Preprocessing":

    st.markdown("## 🔬 Detail Preprocessing")

    if st.session_state.dataset.empty:
        st.warning("⚠️ Upload dataset terlebih dahulu di menu Upload & Training.")
    else:
        df_view = st.session_state.df_tampil if not st.session_state.df_tampil.empty \
                  else st.session_state.dataset
        df=st.session_state.dataset

        st.info("💡 Tabel di bawah menampilkan **semua kolom asli** dataset. Kolom yang dinormalisasi (PENDAPATAN, PEKERJAAN, JUMLAH TANGGUNGAN, STATUS RUMAH, LABEL) sudah diubah ke nilai numerik.")

        df_num = df_view.copy()
        df_num.insert(0,"No",range(1,len(df_num)+1))
        st.markdown("### 📋 Dataset Setelah Preprocessing (Semua Kolom)")
        st.dataframe(df_num, width="stretch", height=420)

        st.markdown("---")
        st.markdown("### 📊 Distribusi Setiap Kolom")

        cols=st.columns(len(df.columns))
        for i, col in enumerate(df.columns):
            with cols[i]:
                vc=df[col].value_counts().sort_index()
                st.markdown(f"<div class='card' style='padding:.9rem;'><div class='card-title' style='font-size:.85rem;'>{col}</div>", unsafe_allow_html=True)
                for val, cnt in vc.items():
                    pct=cnt/len(df)*100
                    if col == 'LABEL':
                        w = "#22c55e" if val == 1 else "#f43f5e"
                        label_txt = f"{val} ({'Layak' if val==1 else 'Tidak Layak'})"
                    else:
                        w = C4
                        label_txt = str(val)
                    st.markdown(f"""
                    <div style='margin-bottom:6px;'>
                        <div style='color:{w};font-weight:700;font-size:.85rem;'>{label_txt}</div>
                        <div style='font-size:.78rem;color:{C6}80;'>{cnt} data ({pct:.1f}%)</div>
                        <div style='height:6px;background:{C2};border-radius:3px;margin-top:3px;'>
                            <div style='height:6px;width:{min(int(pct),100)}%;background:{w};border-radius:3px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🗂 Panduan Encoding Numerik")
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>💰 Pendapatan Orang Tua</div>
                <div style='font-size:.88rem;color:{C6};line-height:2;'>
                <b style='color:{C4};'>0</b> = Rendah : Rp 500rb – Rp 1jt<br>
                <b style='color:{C4};'>1</b> = Sedang : Rp 1jt – Rp 4jt<br>
                <b style='color:{C4};'>2</b> = Tinggi : &gt; Rp 4jt
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>💼 Pekerjaan & 👨‍👩‍👧 Tanggungan</div>
                <div style='font-size:.88rem;color:{C6};line-height:2;'>
                <b style='color:{C4};'>0</b>=Petani · <b style='color:{C4};'>1</b>=PNS<br>
                <b style='color:{C4};'>2</b>=Polisi · <b style='color:{C4};'>3</b>=Wiraswasta<br>
                <b style='color:{C4};'>4</b>=Buruh · dst.<br>
                Tanggungan = angka asli (1,2,3,…)
                </div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>🏠 Status Rumah & Label</div>
                <div style='font-size:.88rem;color:{C6};line-height:2;'>
                <b style='color:{C4};'>0</b> = Kontrak/Sewa<br>
                <b style='color:{C4};'>1</b> = Milik Sendiri<br>
                <b style='color:#22c55e;'>Label 1</b> = Layak BSM<br>
                <b style='color:#f43f5e;'>Label 0</b> = Tidak Layak
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        c_a,c_b=st.columns(2)
        with c_a:
            st.download_button("📥 Download CSV", data=df_to_csv_bytes(df),
                file_name="preprocessing_bsm.csv", mime="text/csv", width="stretch")
        with c_b:
            st.download_button("📥 Download Excel", data=df_to_excel_bytes(df),
                file_name="preprocessing_bsm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch")

# =========================================================
# 🧮 PERHITUNGAN MANUAL
# =========================================================

elif menu == "🧮 Perhitungan Manual":

    st.markdown("## 🧮 Perhitungan Naive Bayes Manual")

    if st.session_state.dataset.empty:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df=st.session_state.dataset
        TARGET='LABEL'; FITUR=[c for c in df.columns if c!=TARGET]
        kelas_list=sorted(df[TARGET].unique().tolist())
        total=len(df); ck=Counter(df[TARGET])

        st.markdown(f"<div class='card'><div class='card-title'><span class='step-badge'>Langkah 1</span> Hitung Prior P(C)</div>", unsafe_allow_html=True)
        cols=st.columns(len(kelas_list)+1)
        with cols[0]:
            st.markdown(f"<div class='metric-pill'><div class='val'>{total}</div><div class='lbl'>Total Data (N)</div></div>", unsafe_allow_html=True)
        for i,k in enumerate(kelas_list):
            p=ck[k]/total; w="#22c55e" if k==1 else "#f43f5e"
            lbl_k = "Ya (Layak)" if k==1 else "Tidak (Tidak Layak)"
            with cols[i+1]:
                st.markdown(f"<div class='metric-pill'><div class='val' style='color:{w};'>{round(p,4)}</div><div class='lbl'>P({lbl_k}) = {ck[k]}/{total}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='card'><div class='card-title'><span class='step-badge'>Langkah 2</span> Hitung Likelihood Gaussian P(xi | C) — Mean & Std Dev</div>", unsafe_allow_html=True)
        st.info("Gaussian Naive Bayes menggunakan **distribusi normal** untuk menghitung P(xi|C). Setiap fitur numerik dianggap berdistribusi Gaussian pada tiap kelas.")
        st.latex(r"P(x_i \mid C) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)")

        if st.session_state.nb_model:
            nb = st.session_state.nb_model
            for f in FITUR:
                st.markdown(f"#### \U0001f539 {f}")
                rows_lh = []
                for k in kelas_list:
                    mu  = round(float(nb.mean_[k][f]), 4)
                    sig = round(float(nb.std_[k][f]),  4)
                    lbl = "Ya (Layak)" if k==1 else "Tidak (Tidak Layak)"
                    rows_lh.append({
                        "Kelas"       : lbl,
                        "Mean (μ)"    : mu,
                        "Std (σ)"     : sig,
                        "Rumus PDF"   : f"1/(√2π × {sig}) × exp(-(x-{mu})²/2×{sig}²)",
                    })
                st.dataframe(pd.DataFrame(rows_lh), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='card'><div class='card-title'><span class='step-badge'>Langkah 3</span> Rumus Posterior & Prediksi</div>", unsafe_allow_html=True)
        st.latex(r"P(C \mid X) \propto P(C) \times \prod_{i=1}^{n} P(x_i \mid C)")
        st.markdown(f"<div style='color:{C6};font-weight:600;font-size:.9rem;margin-bottom:.6rem;'>Untuk efisiensi, digunakan <b>log-posterior</b> agar terhindar dari underflow bilangan kecil :</div>", unsafe_allow_html=True)
        st.latex(r"\log P(C \mid X) \propto \log P(C) + \sum_{{i=1}}^{{n}} \log P(x_i \mid C)")
        st.markdown(f"<div style='color:{C6};font-weight:600;font-size:.9rem;'>Kelas dengan <b>log-posterior terbesar</b> dipilih sebagai hasil prediksi.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### \U0001f50d Contoh Perhitungan Per Data")
        idx_data = st.number_input("Pilih nomor data (baris ke-)", min_value=1, max_value=len(df), value=1, step=1)
        contoh_row = df.iloc[idx_data - 1]
        if st.session_state.nb_model:
            nb = st.session_state.nb_model
            lbl_aktual = "Ya (Layak)" if contoh_row[TARGET]==1 else "Tidak (Tidak Layak)"
            warna_aktual = "#22c55e" if contoh_row[TARGET]==1 else "#f43f5e"
            info_fitur = "  ".join([f"<b>{f}</b>: {contoh_row[f]}" for f in FITUR])
            st.markdown(f"""
            <div class='info-box'>
                <h4>\U0001f4cb Data ke-{idx_data}</h4>
                <p>{info_fitur}<br>
                <b>Label Aktual</b>: <span style='color:{warna_aktual};font-weight:800;'>{lbl_aktual}</span></p>
            </div>
            """, unsafe_allow_html=True)

            rows_post = []
            for k in kelas_list:
                lp  = np.log(nb.prior[k])
                det = f"log({round(nb.prior[k],4)})"
                for f in FITUR:
                    xi  = float(contoh_row[f])
                    mu  = float(nb.mean_[k][f])
                    sig = float(nb.std_[k][f])
                    p   = gaussian_prob(xi, mu, sig)
                    lp += np.log(p + 1e-300)
                    det += f" + log(N({xi:.2f};{mu:.3f},{sig:.3f}))"
                rows_post.append({
                    "Kelas"         : "Ya (Layak)" if k==1 else "Tidak (Tidak Layak)",
                    "Log Posterior" : round(lp, 6),
                    "Detail"        : det,
                })
            df_post = pd.DataFrame(rows_post)
            best_idx = df_post["Log Posterior"].idxmax()
            best_lbl = df_post.loc[best_idx, "Kelas"]
            df_post["Hasil"] = ["\u2705 Terpilih" if i==best_idx else "" for i in df_post.index]
            st.dataframe(df_post, width="stretch", hide_index=True)
            warna_pred = "#22c55e" if "Ya" in best_lbl else "#f43f5e"
            st.markdown(f"<div style='text-align:center;padding:.8rem;'><span style='color:{warna_pred};font-size:1.1rem;font-weight:900;'>\u25b6 Prediksi Data ke-{idx_data} : {best_lbl}</span></div>", unsafe_allow_html=True)

        with st.expander("\U0001f4c4 Lihat Perhitungan Lengkap (Format Terminal)"):
            if st.session_state.nb_model:
                nb  = st.session_state.nb_model
                out = f"{'='*55}\nGAUSSIAN NAIVE BAYES — PERHITUNGAN MANUAL\n{'='*55}\n\n"
                out += f"Total Data = {total}\n\n--- PRIOR P(C) ---\n"
                for k in kelas_list:
                    lbl = "Ya (Layak)" if k==1 else "Tidak (Tidak Layak)"
                    out += f"P({lbl}) = {ck[k]}/{total} = {round(ck[k]/total,6)}\n"
                out += "\n--- MEAN & STD PER FITUR ---\n"
                for k in kelas_list:
                    lbl = "Ya (Layak)" if k==1 else "Tidak (Tidak Layak)"
                    out += f"\nKelas : {lbl}\n"
                    for f in FITUR:
                        mu  = round(float(nb.mean_[k][f]),4)
                        sig = round(float(nb.std_[k][f]),4)
                        out += f"  {f}: mean={mu}, std={sig}\n"
                out += "\n--- RUMUS PDF GAUSSIAN ---\n"
                out += "P(xi|C) = 1/(sqrt(2pi)*sigma) * exp(-(xi-mu)^2 / (2*sigma^2))\n"
                st.code(out, language="text")


# =========================================================
# 📊 EVALUASI & GRAFIK
# =========================================================

elif menu == "📊 Evaluasi & Grafik":

    st.markdown("## 📊 Evaluasi Model Naive Bayes")

    if st.session_state.dataset.empty or st.session_state.nb_model is None:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        model  = st.session_state.nb_model
        df_tr  = st.session_state.df_train if not st.session_state.df_train.empty else st.session_state.dataset
        df_te  = st.session_state.df_test
        TARGET = 'LABEL'
        FITUR  = [c for c in df_tr.columns if c != TARGET]

        yt_tr  = df_tr[TARGET].tolist(); yp_tr = model.predict(df_tr[FITUR])
        yt_te  = df_te[TARGET].tolist() if not df_te.empty else []
        yp_te  = model.predict(df_te[FITUR]) if not df_te.empty else []
        labels = sorted(set(yt_tr))

        acc_tr = accuracy_score(yt_tr, yp_tr)
        acc_te = accuracy_score(yt_te, yp_te) if yt_te else 0

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{C2},{C5});border:2px solid {C3}60;
            border-radius:18px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;
            box-shadow:0 4px 16px {C3}20;'>
            <div style='font-size:.95rem;font-weight:900;color:{C4};margin-bottom:.8rem;'>
                📊 Perbandingan Akurasi Data Latih vs Data Uji
            </div>
            <div style='display:flex;gap:1rem;flex-wrap:wrap;'>
                <div style='flex:1;background:{C5};border:2px solid #22c55e;border-radius:12px;
                    padding:.8rem 1rem;text-align:center;min-width:130px;'>
                    <div style='font-size:1.7rem;font-weight:900;color:#22c55e;'>{round(acc_tr*100,2)}%</div>
                    <div style='font-size:.78rem;font-weight:700;color:{C6}80;'>Akurasi Data Latih (80%)</div>
                    <div style='font-size:.75rem;color:{C6}60;'>{len(df_tr)} data</div>
                </div>
                <div style='flex:1;background:{C5};border:2px solid #f59e0b;border-radius:12px;
                    padding:.8rem 1rem;text-align:center;min-width:130px;'>
                    <div style='font-size:1.7rem;font-weight:900;color:#f59e0b;'>{round(acc_te*100,2)}%</div>
                    <div style='font-size:.78rem;font-weight:700;color:{C6}80;'>Akurasi Data Uji (20%)</div>
                    <div style='font-size:.75rem;color:{C6}60;'>{len(df_te)} data</div>
                </div>
                <div style='flex:1;background:{C5};border:1.5px solid {C3}50;border-radius:12px;
                    padding:.8rem 1rem;text-align:center;min-width:130px;'>
                    <div style='font-size:1.7rem;font-weight:900;color:{C4};'>{len(df_tr)+len(df_te)}</div>
                    <div style='font-size:.78rem;font-weight:700;color:{C6}80;'>Total Dataset</div>
                    <div style='font-size:.75rem;color:{C6}60;'>80% latih + 20% uji</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_eval_tr, tab_eval_te = st.tabs(["🟢 Evaluasi Data Latih (80%)", "🟡 Evaluasi Data Uji (20%)"])

        # ---- FUNGSI EVALUASI DENGAN KEY UNIK PER TAB ----
        def tampil_evaluasi(y_true, y_pred, labels, judul_set, key_prefix):
            acc  = accuracy_score(y_true, y_pred)
            cm   = confusion_matrix(y_true, y_pred, labels=labels)
            benar = sum(a==p for a,p in zip(y_true,y_pred)); salah=len(y_true)-benar

            st.markdown(f"### 📈 Metrik Utama — {judul_set}")
            c1,c2,c3,c4 = st.columns(4)
            for col, val, lbl_s, warna in [
                (c1, f"{round(acc*100,2)}%", "Akurasi", C4),
                (c2, benar, "Prediksi Benar", "#22c55e"),
                (c3, salah, "Prediksi Salah", "#f43f5e"),
                (c4, len(y_true), "Total Data", C4),
            ]:
                with col:
                    st.markdown(f"<div class='metric-pill'><div class='val' style='color:{warna};'>{val}</div><div class='lbl'>{lbl_s}</div></div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🗂 Confusion Matrix")
            c_kiri, c_kanan = st.columns(2)
            with c_kiri:
                fig_cm=ff.create_annotated_heatmap(
                    z=cm.tolist(),
                    x=[f"Pred : {l}" for l in labels],
                    y=[f"Aktual : {l}" for l in labels],
                    annotation_text=[[str(v) for v in row] for row in cm.tolist()],
                    colorscale="RdPu", showscale=True,
                )
                fig_cm.update_layout(title=dict(text=f"Confusion Matrix — {judul_set}",font=dict(size=14)),
                    height=350, **PLOT_LAYOUT)
                # KEY UNIK: gunakan key_prefix
                st.plotly_chart(fig_cm, width="stretch", key=f"{key_prefix}_cm")

            with c_kanan:
                if len(labels)==2:
                    tn,fp,fn,tp=cm.ravel()
                    fig_bar=go.Figure(data=[go.Bar(
                        x=["TP","TN","FP","FN"], y=[tp,tn,fp,fn],
                        text=[tp,tn,fp,fn], textposition="outside",
                        marker_color=["#22c55e","#22c55e","#f43f5e","#f43f5e"],
                        marker_line_width=0,
                    )])
                    fig_bar.update_layout(title=dict(text="TP / TN / FP / FN",font=dict(size=14)),
                        height=350, yaxis=dict(gridcolor="#e5e7eb"), showlegend=False, **PLOT_LAYOUT)
                    # KEY UNIK
                    st.plotly_chart(fig_bar, width="stretch", key=f"{key_prefix}_bar_cm")
                    st.markdown(f"""
                    <div style='font-size:.88rem;line-height:2.2;color:{C6};
                         background:{C2};border-radius:12px;padding:.8rem 1rem;'>
                    <span style='color:#22c55e;font-weight:800;'>● TP = {tp}</span> → Pred Ya & Aktual Ya<br>
                    <span style='color:#22c55e;font-weight:800;'>● TN = {tn}</span> → Pred Tidak & Aktual Tidak<br>
                    <span style='color:#f43f5e;font-weight:800;'>● FP = {fp}</span> → Pred Ya, Aktual Tidak<br>
                    <span style='color:#f43f5e;font-weight:800;'>● FN = {fn}</span> → Pred Tidak, Aktual Ya
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Grafik Evaluasi")
            ca,cb,cc = st.columns(3)
            with ca:
                fig_pie=go.Figure(data=[go.Pie(
                    labels=["Benar","Salah"], values=[benar,salah], hole=.5,
                    marker_colors=["#22c55e","#f43f5e"],
                    textinfo="label+percent", textfont_size=13,
                )])
                fig_pie.update_layout(title=dict(text="Akurasi Prediksi",font=dict(size=14)),
                    height=300, showlegend=False, **PLOT_LAYOUT)
                # KEY UNIK
                st.plotly_chart(fig_pie, width="stretch", key=f"{key_prefix}_pie_akurasi")
            with cb:
                ck_eval=Counter(y_true)
                fig_dist=go.Figure(data=[go.Bar(
                    x=list(ck_eval.keys()), y=list(ck_eval.values()),
                    text=list(ck_eval.values()), textposition="outside",
                    marker_color=["#22c55e" if k==1 else "#f43f5e" for k in ck_eval],
                )])
                fig_dist.update_layout(title=dict(text="Distribusi Kelas Aktual",font=dict(size=14)),
                    height=300, yaxis=dict(gridcolor="#e5e7eb"), showlegend=False, **PLOT_LAYOUT)
                # KEY UNIK
                st.plotly_chart(fig_dist, width="stretch", key=f"{key_prefix}_bar_aktual")
            with cc:
                ck_pred=Counter(y_pred)
                fig_pr=go.Figure(data=[go.Bar(
                    x=list(ck_pred.keys()), y=list(ck_pred.values()),
                    text=list(ck_pred.values()), textposition="outside",
                    marker_color=[C3]*len(ck_pred),
                )])
                fig_pr.update_layout(title=dict(text="Distribusi Kelas Prediksi",font=dict(size=14)),
                    height=300, yaxis=dict(gridcolor="#e5e7eb"), showlegend=False, **PLOT_LAYOUT)
                # KEY UNIK
                st.plotly_chart(fig_pr, width="stretch", key=f"{key_prefix}_bar_pred")

            st.markdown("---")
            st.markdown("### 📋 Classification Report")
            c_l2, c_r2 = st.columns(2)
            with c_l2:
                rpt=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True)).transpose().round(4)
                st.dataframe(rpt, width="stretch")
            with c_r2:
                rpt_clean={k:v for k,v in classification_report(y_true,y_pred,output_dict=True).items() if k in labels}
                fig_rpt=go.Figure()
                for m, warna_m in [("precision",C3),("recall",C4),("f1-score",C8)]:
                    fig_rpt.add_trace(go.Bar(
                        name=m.capitalize(),
                        x=list(rpt_clean.keys()),
                        y=[rpt_clean[k][m] for k in rpt_clean],
                        marker_color=warna_m,
                        text=[round(rpt_clean[k][m],3) for k in rpt_clean],
                        textposition="outside",
                    ))
                fig_rpt.update_layout(barmode="group",
                    title=dict(text="Precision / Recall / F1-Score",font=dict(size=14)),
                    height=320, yaxis=dict(gridcolor="#e5e7eb",range=[0,1.2]),
                    legend=dict(orientation="h",y=1.18), **PLOT_LAYOUT)
                # KEY UNIK
                st.plotly_chart(fig_rpt, width="stretch", key=f"{key_prefix}_bar_clf")

        with tab_eval_tr:
            # key_prefix = "tr" untuk data latih
            tampil_evaluasi(yt_tr, yp_tr, labels, "Data Latih 80%", key_prefix="tr")
            st.markdown("---")
            st.markdown("### 🔎 Detail Prediksi — Data Latih")
            df_hasil_tr = df_tr.copy()
            df_hasil_tr["Prediksi"] = yp_tr
            df_hasil_tr["Status"]   = ["✅ Benar" if a==p else "❌ Salah" for a,p in zip(yt_tr,yp_tr)]
            df_hasil_tr.insert(0,"No",range(1,len(df_hasil_tr)+1))
            st.dataframe(df_hasil_tr, width="stretch", height=380)
            c_d1,c_d2 = st.columns(2)
            with c_d1:
                st.download_button("📥 CSV — Hasil Data Latih",
                    data=df_to_csv_bytes(df_hasil_tr), file_name="evaluasi_data_latih.csv",
                    mime="text/csv", width="stretch")
            with c_d2:
                st.download_button("📥 Excel — Hasil Data Latih",
                    data=df_to_excel_bytes(df_hasil_tr), file_name="evaluasi_data_latih.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch")

        with tab_eval_te:
            if not df_te.empty:
                # key_prefix = "te" untuk data uji
                tampil_evaluasi(yt_te, yp_te, labels, "Data Uji 20%", key_prefix="te")
                st.markdown("---")
                st.markdown("### 🔎 Detail Prediksi — Data Uji")
                df_hasil_te = df_te.copy()
                df_hasil_te["Prediksi"] = yp_te
                df_hasil_te["Status"]   = ["✅ Benar" if a==p else "❌ Salah" for a,p in zip(yt_te,yp_te)]
                df_hasil_te.insert(0,"No",range(1,len(df_hasil_te)+1))
                st.dataframe(df_hasil_te, width="stretch", height=380)
                c_d1,c_d2 = st.columns(2)
                with c_d1:
                    st.download_button("📥 CSV — Hasil Data Uji",
                        data=df_to_csv_bytes(df_hasil_te), file_name="evaluasi_data_uji.csv",
                        mime="text/csv", width="stretch")
                with c_d2:
                    st.download_button("📥 Excel — Hasil Data Uji",
                        data=df_to_excel_bytes(df_hasil_te), file_name="evaluasi_data_uji.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch")
            else:
                st.info("💡 Data uji 20% akan tersedia setelah upload dataset.")

# =========================================================
# 🔍 PREDIKSI DATA BARU
# =========================================================

elif menu == "🔍 Prediksi Data Baru":

    st.markdown("## 🔍 Prediksi Data Baru")

    if st.session_state.nb_model is None:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        model=st.session_state.nb_model
        df_ref=st.session_state.dataset

        # Mapping teks → kode numerik
        _map_pend  = {"Rendah (0)":0, "Sedang (1)":1, "Tinggi (2)":2}
        _map_rumah = {"Kontrak / Sewa (0)":0, "Milik Sendiri (1)":1}
        _map_pek   = {
            "Petani (0)":0, "PNS - Pegawai Negeri Sipil (1)":1,
            "Polisi (2)":2, "Wiraswasta (3)":3,
            "Buruh (4)":4, "Pedagang (5)":5,
            "Nelayan (6)":6, "Swasta (7)":7,
            "Honorer (8)":8, "Tidak Bekerja (9)":9,
        }

        st.markdown(f"<div class='card'><div class='card-title'>✏️ Masukkan Data Siswa</div>", unsafe_allow_html=True)

        # -- Kotak keterangan kode --
        with st.expander("📌 Lihat Keterangan Kode Numerik Setiap Variabel"):
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_keterangan("Pendapatan Orang Tua", KETERANGAN_PENDAPATAN)
            with k2:
                render_keterangan("Pekerjaan Orang Tua", KETERANGAN_PEKERJAAN)
            with k3:
                render_keterangan("Status Rumah", KETERANGAN_RUMAH)
            with k4:
                render_keterangan("Label / Hasil", KETERANGAN_LABEL)

        c1,c2=st.columns(2)
        with c1:
            pendapatan = st.selectbox(
                "💰 Pendapatan Orang Tua",
                list(_map_pend.keys()),
                help="0=Rendah (≤Rp1jt) | 1=Sedang (Rp1jt–4jt) | 2=Tinggi (>Rp4jt)"
            )
            tanggungan_input = st.number_input(
                "👨‍👩‍👧 Jumlah Tanggungan (angka)",
                min_value=1, max_value=6, value=3, step=1,
                help="Masukkan jumlah anggota keluarga yang ditanggung (angka asli, misal: 3)"
            )
        with c2:
            pekerjaan = st.selectbox(
                "💼 Pekerjaan Orang Tua",
                list(_map_pek.keys()),
                help="Pilih pekerjaan yang sesuai. Kode dalam kurung adalah nilai numeriknya."
            )
            rumah = st.selectbox(
                "🏠 Status Rumah",
                list(_map_rumah.keys()),
                help="0=Kontrak/Sewa | 1=Milik Sendiri"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔍 Prediksi Sekarang", width="stretch"):
            pend_num = _map_pend[pendapatan]
            pek_num  = _map_pek[pekerjaan]
            tang_num = float(tanggungan_input)
            rmh_num  = _map_rumah[rumah]

            row_num  = {
                "PENDAPATAN ORANG TUA" : pend_num,
                "PEKERJAAN ORANG TUA"  : pek_num,
                "JUMLAH TANGGUNGAN"    : tang_num,
                "STATUS RUMAH"         : rmh_num,
            }

            proba  = model.predict_proba_log(row_num)
            hasil  = model.predict_one(row_num)
            hasil_lbl = "Ya" if hasil==1 else "Tidak"

            st.markdown("---")
            if hasil_lbl=="Ya":
                st.markdown("<div style='text-align:center;padding:1.2rem;'><span class='badge-ya'>✅ LAYAK MENERIMA BSM</span></div>", unsafe_allow_html=True)
                st.success("Siswa ini **LAYAK** menerima Bantuan Siswa Miskin (BSM).")
            else:
                st.markdown("<div style='text-align:center;padding:1.2rem;'><span class='badge-tidak'>❌ TIDAK LAYAK MENERIMA BSM</span></div>", unsafe_allow_html=True)
                st.error("Siswa ini **TIDAK LAYAK** menerima Bantuan Siswa Miskin (BSM).")

            ca,cb=st.columns(2)
            with ca:
                st.markdown(f"<div class='card'><div class='card-title'>📋 Data Input (Numerik)</div>", unsafe_allow_html=True)
                input_info = [
                    ("💰 Pendapatan", f"{pendapatan} → kode {pend_num}"),
                    ("💼 Pekerjaan",  f"{pekerjaan} → kode {pek_num}"),
                    ("👨‍👩‍👧 Tanggungan", f"{int(tang_num)} orang"),
                    ("🏠 Status Rumah", f"{rumah} → kode {rmh_num}"),
                ]
                for label_i, val_i in input_info:
                    st.markdown(f"<div style='font-size:.9rem;color:{C6};padding:3px 0;'><b>{label_i}</b> : {val_i}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with cb:
                st.markdown(f"<div class='card'><div class='card-title'>🧮 Log Posterior</div>", unsafe_allow_html=True)
                for k,v in proba.items():
                    lbl_k = "Ya (Layak)" if k==1 else "Tidak (Tidak Layak)"
                    w="#22c55e" if k==hasil else "#f43f5e"
                    terpilih=" ← Terpilih" if k==hasil else ""
                    st.markdown(f"<div style='color:{w};font-weight:700;font-size:.9rem;padding:3px 0;'>Kelas {lbl_k} : {round(v,6)}{terpilih}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 📝 DATA UJI
# =========================================================

elif menu == "📝 Data Uji":

    st.markdown("## 📝 Data Uji")

    if st.session_state.nb_model is None:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        model=st.session_state.nb_model
        df_ref=st.session_state.dataset
        TARGET='LABEL'; FITUR=[c for c in df_ref.columns if c!=TARGET]

        # Mapping dengan label deskriptif
        _map_pend_uji  = {"Rendah (0)":0, "Sedang (1)":1, "Tinggi (2)":2}
        _map_rumah_uji = {"Kontrak / Sewa (0)":0, "Milik Sendiri (1)":1}
        _map_pek_uji   = {
            "Petani (0)":0, "PNS - Pegawai Negeri Sipil (1)":1,
            "Polisi (2)":2, "Wiraswasta (3)":3,
            "Buruh (4)":4, "Pedagang (5)":5,
            "Nelayan (6)":6, "Swasta (7)":7,
            "Honorer (8)":8, "Tidak Bekerja (9)":9,
        }

        tab1, tab2, tab3 = st.tabs(["➕ Tambah Manual", "📂 Import File", "📊 Hasil Uji"])

        # TAB 1 — TAMBAH MANUAL
        with tab1:
            st.markdown(f"<div class='card'><div class='card-title'>✏️ Input Data Baru</div>", unsafe_allow_html=True)

            # Keterangan kode
            with st.expander("📌 Lihat Keterangan Kode Setiap Variabel"):
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    render_keterangan("Pendapatan Orang Tua", KETERANGAN_PENDAPATAN)
                with k2:
                    render_keterangan("Pekerjaan Orang Tua", KETERANGAN_PEKERJAAN)
                with k3:
                    render_keterangan("Status Rumah", KETERANGAN_RUMAH)
                with k4:
                    render_keterangan("Label / Hasil", KETERANGAN_LABEL)

            c1,c2=st.columns(2)
            with c1:
                t_pend = st.selectbox(
                    "💰 Pendapatan Orang Tua",
                    list(_map_pend_uji.keys()),
                    key="du1",
                    help="0=Rendah | 1=Sedang | 2=Tinggi"
                )
                t_tang = st.number_input(
                    "👨‍👩‍👧 Jumlah Tanggungan (angka)",
                    min_value=1, max_value=6, value=3, step=1,
                    key="du2",
                    help="Masukkan angka asli jumlah tanggungan, misal: 2, 3, 5"
                )
            with c2:
                t_pek = st.selectbox(
                    "💼 Pekerjaan Orang Tua",
                    list(_map_pek_uji.keys()),
                    key="du3",
                    help="Pilih pekerjaan. Kode dalam kurung adalah nilai numeriknya."
                )
                t_rmh = st.selectbox(
                    "🏠 Status Rumah",
                    list(_map_rumah_uji.keys()),
                    key="du4",
                    help="0=Kontrak/Sewa | 1=Milik Sendiri"
                )
            st.markdown("</div>", unsafe_allow_html=True)

            ca,cb,cc=st.columns(3)
            with ca:
                if st.button("🔍 Prediksi & Tambah", width="stretch"):
                    pend_num_u = _map_pend_uji[t_pend]
                    pek_num_u  = _map_pek_uji[t_pek]
                    tang_val_u = float(t_tang)
                    rmh_num_u  = _map_rumah_uji[t_rmh]

                    row_num_u = {
                        "PENDAPATAN ORANG TUA": pend_num_u,
                        "PEKERJAAN ORANG TUA" : pek_num_u,
                        "JUMLAH TANGGUNGAN"   : tang_val_u,
                        "STATUS RUMAH"        : rmh_num_u,
                    }
                    hasil_num = model.predict_one(row_num_u)
                    hasil_lbl = "Ya (Layak)" if hasil_num==1 else "Tidak (Tidak Layak)"

                    row_simpan = {
                        "PENDAPATAN ORANG TUA": t_pend,
                        "PEKERJAAN ORANG TUA" : t_pek,
                        "JUMLAH TANGGUNGAN"   : int(t_tang),
                        "STATUS RUMAH"        : t_rmh,
                        TARGET                : hasil_lbl,
                    }
                    new_row=pd.DataFrame([row_simpan])
                    st.session_state.data_uji=pd.concat([st.session_state.data_uji, new_row], ignore_index=True)
                    if hasil_num==1:
                        st.success(f"✅ Ditambahkan — Prediksi: **{hasil_lbl}**")
                    else:
                        st.error(f"❌ Ditambahkan — Prediksi: **{hasil_lbl}**")

            with cb:
                if st.button("🗑 Hapus Data Terakhir", width="stretch"):
                    if len(st.session_state.data_uji)>0:
                        st.session_state.data_uji=st.session_state.data_uji.iloc[:-1].reset_index(drop=True)
                        st.warning("Data terakhir dihapus.")

            with cc:
                if st.button("🗑 Hapus Semua", width="stretch"):
                    st.session_state.data_uji=pd.DataFrame()
                    st.warning("Semua data uji dihapus.")

        # TAB 2 — IMPORT FILE
        with tab2:
            st.info("Upload file data uji (tanpa kolom LABEL). Sistem akan memprediksi otomatis.")
            file_uji=st.file_uploader("Upload File Data Uji", type=["csv","xlsx"], key="fu")

            if file_uji and st.button("📂 Import & Prediksi Semua"):
                try:
                    df_imp=baca_file(file_uji)
                    _, df_clean = auto_preprocess(df_imp)

                    if 'LABEL' not in df_clean.columns:
                        fitur_ada=[f for f in FITUR if f in df_clean.columns]
                        y_imp = model.predict(df_clean[fitur_ada])
                        df_clean['LABEL'] = ["Ya (Layak)" if v==1 else "Tidak (Tidak Layak)" for v in y_imp]

                    st.session_state.data_uji=pd.concat(
                        [st.session_state.data_uji, df_clean], ignore_index=True)
                    st.success(f"✅ {len(df_clean)} data berhasil diimport & diprediksi.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        # TAB 3 — HASIL UJI
        with tab3:
            if st.session_state.data_uji.empty:
                st.info("Belum ada data uji. Tambahkan dari tab ➕ atau 📂")
            else:
                df_u=st.session_state.data_uji.copy()
                df_u.insert(0,"No",range(1,len(df_u)+1))
                st.markdown(f"**Total data uji : {len(df_u)}**")
                st.dataframe(df_u, width="stretch", height=420)

                st.session_state.hasil_uji=st.session_state.data_uji.copy()

                st.markdown("---")
                ca,cb=st.columns(2)
                with ca:
                    st.download_button("📥 Download Data Uji CSV",
                        data=df_to_csv_bytes(df_u), file_name="data_uji_bsm.csv",
                        mime="text/csv", width="stretch")
                with cb:
                    st.download_button("📥 Download Data Uji Excel",
                        data=df_to_excel_bytes(df_u), file_name="data_uji_bsm.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch")

# =========================================================
# 💾 UNDUH HASIL
# =========================================================

elif menu == "💾 Unduh Hasil":

    st.markdown("## 💾 Unduh Semua Hasil")

    if st.session_state.dataset.empty:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df=st.session_state.dataset
        model=st.session_state.nb_model

        st.markdown(f"""
        <div class='info-box'>
            <h4>📌 Panduan Download</h4>
            <p>Semua file hasil analisis tersedia di bawah ini.
            Klik tombol untuk mengunduh dalam format CSV atau Excel.</p>
        </div>
        """, unsafe_allow_html=True)

        df_tampil_dl = st.session_state.df_tampil if not st.session_state.df_tampil.empty else df
        st.markdown(f"<div class='card'><div class='card-title'>🔬 1. Dataset Hasil Preprocessing</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:.88rem;color:{C6};margin-bottom:.8rem;'>Dataset setelah normalisasi — {len(df_tampil_dl)} data, {len(df_tampil_dl.columns)} kolom.</div>", unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.download_button("📥 CSV — Preprocessing",
                data=df_to_csv_bytes(df_tampil_dl), file_name="01_preprocessing_bsm.csv",
                mime="text/csv", width="stretch")
        with c2:
            st.download_button("📥 Excel — Preprocessing",
                data=df_to_excel_bytes(df_tampil_dl), file_name="01_preprocessing_bsm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        if model:
            y_pred=model.predict(df.drop(columns=['LABEL']))
            y_true=df['LABEL'].tolist()
            df_pred=df.copy()
            df_pred["Prediksi"]=y_pred
            df_pred["Status"]=["✅ Benar" if a==p else "❌ Salah" for a,p in zip(y_true,y_pred)]
            df_pred.insert(0,"No",range(1,len(df_pred)+1))
            acc=accuracy_score(y_true,y_pred)

            st.markdown(f"<div class='card'><div class='card-title'>📊 2. Hasil Prediksi Dataset Training</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:.88rem;color:{C6};margin-bottom:.8rem;'>Akurasi : <b style='color:{C4};'>{round(acc*100,2)}%</b> — {sum(a==p for a,p in zip(y_true,y_pred))} benar dari {len(y_true)} data.</div>", unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1:
                st.download_button("📥 CSV — Hasil Prediksi Training",
                    data=df_to_csv_bytes(df_pred), file_name="02_hasil_prediksi_training.csv",
                    mime="text/csv", width="stretch")
            with c2:
                st.download_button("📥 Excel — Hasil Prediksi Training",
                    data=df_to_excel_bytes(df_pred), file_name="02_hasil_prediksi_training.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"<div class='card'><div class='card-title'>📋 3. Laporan Evaluasi Model</div>", unsafe_allow_html=True)
            labels=sorted(set(y_true))
            cm=confusion_matrix(y_true,y_pred,labels=labels)
            rpt=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True)).transpose().round(4)
            cm_df=pd.DataFrame(cm,
                index=[f"Aktual_{l}" for l in labels],
                columns=[f"Prediksi_{l}" for l in labels])
            ringkasan=pd.DataFrame({
                "Metrik":["Akurasi","Total Data","Prediksi Benar","Prediksi Salah"],
                "Nilai":[f"{round(acc*100,2)}%", len(y_true),
                         sum(a==p for a,p in zip(y_true,y_pred)),
                         sum(a!=p for a,p in zip(y_true,y_pred))]
            })
            buf=io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                ringkasan.to_excel(writer, sheet_name='Ringkasan', index=False)
                rpt.to_excel(writer, sheet_name='Classification_Report')
                cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
                df_pred.to_excel(writer, sheet_name='Detail_Prediksi', index=False)
            c1,c2=st.columns(2)
            with c1:
                st.download_button("📥 CSV — Classification Report",
                    data=df_to_csv_bytes(rpt.reset_index()),
                    file_name="03_classification_report.csv",
                    mime="text/csv", width="stretch")
            with c2:
                st.download_button("📥 Excel — Laporan Lengkap (Multi-Sheet)",
                    data=buf.getvalue(), file_name="03_laporan_evaluasi_lengkap.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        if not st.session_state.data_uji.empty:
            df_u=st.session_state.data_uji.copy()
            df_u.insert(0,"No",range(1,len(df_u)+1))
            st.markdown(f"<div class='card'><div class='card-title'>📝 4. Hasil Data Uji</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:.88rem;color:{C6};margin-bottom:.8rem;'>{len(df_u)} data uji tersedia.</div>", unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1:
                st.download_button("📥 CSV — Data Uji",
                    data=df_to_csv_bytes(df_u), file_name="04_data_uji_bsm.csv",
                    mime="text/csv", width="stretch")
            with c2:
                st.download_button("📥 Excel — Data Uji",
                    data=df_to_excel_bytes(df_u), file_name="04_data_uji_bsm.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("💡 Belum ada data uji. Tambahkan di menu **📝 Data Uji** untuk mengaktifkan download ini.")

# =========================================================
# ℹ️ TENTANG SISTEM
# =========================================================

elif menu == "ℹ️ Tentang Sistem":

    st.markdown("## ℹ️ Tentang Sistem")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{C4} 0%,{C6} 100%);
         border-radius:22px;padding:2.5rem 2rem;margin-bottom:1.8rem;text-align:center;
         box-shadow:0 8px 32px {C3}50;'>
        <div style='font-size:3rem;margin-bottom:.5rem;'>{ICON}</div>
        <div style='font-size:1.7rem;font-weight:900;color:white;letter-spacing:.5px;'>
            Sistem Prediksi Penerima BSM
        </div>
        <div style='color:white;opacity:.85;font-size:1rem;font-weight:600;margin-top:.4rem;'>
            Berbasis Algoritma Naive Bayes · Bantuan Siswa Miskin
        </div>
        <div style='margin-top:1.2rem;'>
            <span style='background:rgba(255,255,255,.2);color:white;border-radius:30px;
                padding:.35rem 1rem;font-size:.82rem;font-weight:700;margin:.2rem;display:inline-block;'>
                📌 Versi 1.0
            </span>
            <span style='background:rgba(255,255,255,.2);color:white;border-radius:30px;
                padding:.35rem 1rem;font-size:.82rem;font-weight:700;margin:.2rem;display:inline-block;'>
                🐍 Python · Streamlit
            </span>
            <span style='background:rgba(255,255,255,.2);color:white;border-radius:30px;
                padding:.35rem 1rem;font-size:.82rem;font-weight:700;margin:.2rem;display:inline-block;'>
                🤖 Naive Bayes
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>📖 Deskripsi Sistem</div>
            <div style='font-size:.93rem;color:{C6};line-height:2;'>
            Sistem ini dirancang untuk membantu pihak sekolah dalam
            menentukan kelayakan siswa sebagai penerima
            <b style='color:{C4};'>Bantuan Siswa Miskin (BSM)</b>
            secara objektif dan transparan.<br><br>
            Proses klasifikasi dilakukan menggunakan algoritma
            <b style='color:{C4};'>Naive Bayes</b> yang diimplementasikan
            secara <i>manual</i> tanpa library machine learning eksternal,
            sehingga setiap langkah perhitungan dapat ditelusuri dan dipelajari.<br><br>
            Sistem menerima data dalam format <b>CSV</b> atau <b>Excel (.xlsx)</b>,
            melakukan preprocessing otomatis, melatih model, lalu menghasilkan
            prediksi beserta laporan evaluasi lengkap.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>✨ Fitur Unggulan</div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:.6rem;'>
        """, unsafe_allow_html=True)
        fitur_list = [
            ("📂", "Upload CSV & Excel", "Deteksi header otomatis"),
            ("🔬", "Preprocessing Cerdas", "Normalisasi data kategorik"),
            ("🧮", "Perhitungan Transparan", "Tampilkan prior & likelihood"),
            ("📊", "Evaluasi Lengkap", "Confusion matrix & classification report"),
            ("🔍", "Prediksi Real-time", "Input manual satu siswa"),
            ("📝", "Kelola Data Uji", "Tambah manual atau import file"),
            ("💾", "Ekspor Multi-format", "Download CSV & Excel multi-sheet"),
            ("📈", "Visualisasi Interaktif", "Grafik berbasis Plotly"),
        ]
        for ikon, judul, keterangan in fitur_list:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{C2},{C5});
                border:1.5px solid {C3}50;border-radius:14px;padding:.75rem 1rem;'>
                <div style='font-size:1.3rem;'>{ikon}</div>
                <div style='font-weight:800;color:{C4};font-size:.88rem;margin-top:.2rem;'>{judul}</div>
                <div style='font-size:.78rem;color:{C6}80;font-weight:600;'>{keterangan}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
        <div class='card' style='text-align:center;'>
            <div class='card-title'>👩‍💻 Pengembang</div>
            <div style='font-size:3rem;margin:.5rem 0;'>🎀</div>
            <div style='font-size:1.1rem;font-weight:900;color:{C4};'>Mahasiswa</div>
            <div style='font-size:.85rem;color:{C6}80;font-weight:600;margin-top:.2rem;'>
                Program Studi Informatika
            </div>
            <div style='margin-top:1rem;padding-top:1rem;border-top:1.5px solid {C3}40;'>
                <div style='font-size:.82rem;font-weight:700;color:{C6};line-height:2;'>
                    🏛 Universitas / Institusi<br>
                    📅 Tahun Akademik 2024/2025<br>
                    🎯 Skripsi / Tugas Akhir
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>🛠 Teknologi</div>
        """, unsafe_allow_html=True)
        teknologi = [
            ("🐍", "Python 3.x", C4),
            ("⚡", "Streamlit", C4),
            ("🐼", "Pandas", "#22c55e"),
            ("🔢", "NumPy", "#22c55e"),
            ("📊", "Plotly", "#f59e0b"),
            ("📋", "Scikit-learn (evaluasi)", "#8b5cf6"),
            ("📄", "OpenPyXL", C6),
        ]
        for ikon, nama, warna in teknologi:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:.6rem;padding:.35rem 0;
                border-bottom:1px solid {C3}20;'>
                <span style='font-size:1rem;'>{ikon}</span>
                <span style='font-weight:700;color:{warna};font-size:.88rem;'>{nama}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>📌 Atribut Klasifikasi</div>
            <div style='font-size:.85rem;color:{C6};line-height:2.1;'>
            <span style='color:{C4};font-weight:800;'>1.</span> Pendapatan Orang Tua<br>
            <span style='color:{C4};font-weight:800;'>2.</span> Pekerjaan Orang Tua<br>
            <span style='color:{C4};font-weight:800;'>3.</span> Jumlah Tanggungan<br>
            <span style='color:{C4};font-weight:800;'>4.</span> Status Rumah<br>
            <span style='color:#22c55e;font-weight:800;'>→</span>
            <b>Label :</b> 1=Ya (Layak) / 0=Tidak
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div class='card-title' style='font-size:1.1rem;color:{C4};font-weight:900;padding:.5rem 0;'>🗺 Alur Penggunaan Sistem</div>", unsafe_allow_html=True)

    langkah = [
        ("📂", "Upload Dataset", "Masuk ke menu Upload & Training. Unggah file CSV atau Excel berisi data siswa beserta labelnya (Ya/Tidak)."),
        ("🔬", "Cek Preprocessing", "Buka menu Preprocessing untuk melihat hasil normalisasi data — pendapatan, pekerjaan, tanggungan, dan status rumah."),
        ("🧮", "Pelajari Perhitungan", "Di menu Perhitungan Manual, pelajari bagaimana nilai Prior dan Likelihood dihitung secara transparan langkah demi langkah."),
        ("📊", "Evaluasi Model", "Periksa performa model di menu Evaluasi & Grafik — akurasi, confusion matrix, precision, recall, dan F1-Score."),
        ("🔍", "Prediksi Siswa Baru", "Gunakan menu Prediksi Data Baru untuk memasukkan data satu siswa dan mendapatkan hasil prediksi kelayakan BSM."),
        ("📝", "Kelola Data Uji", "Tambahkan banyak data uji secara manual atau import file di menu Data Uji, lalu lihat seluruh hasilnya sekaligus."),
        ("💾", "Download Hasil", "Unduh semua laporan dalam format CSV atau Excel multi-sheet dari menu Unduh Hasil."),
    ]

    cols = st.columns(len(langkah))
    for i, (ikon, judul, desk) in enumerate(langkah):
        with cols[i]:
            st.markdown(f"""
            <div style='background:linear-gradient(180deg,{C2} 0%,{C5} 100%);
                border:1.5px solid {C3}50;border-radius:16px;padding:1rem .8rem;
                text-align:center;height:100%;box-shadow:0 3px 12px {C3}15;'>
                <div style='width:32px;height:32px;background:linear-gradient(135deg,{C3},{C4});
                    border-radius:50%;display:flex;align-items:center;justify-content:center;
                    margin:0 auto .5rem;color:white;font-weight:900;font-size:.85rem;'>{i+1}</div>
                <div style='font-size:1.4rem;margin-bottom:.3rem;'>{ikon}</div>
                <div style='font-weight:900;color:{C4};font-size:.85rem;margin-bottom:.4rem;'>{judul}</div>
                <div style='font-size:.75rem;color:{C6}80;font-weight:600;line-height:1.6;'>{desk}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center;padding:1.2rem;background:linear-gradient(135deg,{C2},{C5});
        border:1.5px solid {C3}40;border-radius:16px;'>
        <div style='font-size:.85rem;font-weight:700;color:{C6}80;'>
            {ICON} Sistem Prediksi Penerima BSM · Naive Bayes · 2025
        </div>
        <div style='font-size:.78rem;color:{C6}50;margin-top:.3rem;font-weight:600;'>
            Dibuat sebagai bagian dari penelitian skripsi · Hak cipta dilindungi
        </div>
    </div>
    """, unsafe_allow_html=True)
