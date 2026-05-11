# =========================================================
# APLIKASI NAIVE BAYES - KELAYAKAN PENERIMA BSM
# STREAMLIT - UI/UX CANTIK BIRU MUDA / PINK MUDA
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
import os
import io
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

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
# TEMA WARNA — ganti ke "pink" atau "biru"
# =========================================================

TEMA = "pink"   # ganti ke "biru" jika ingin tema biru

if TEMA == "pink":
    C1  = "#f8e8f0"   # latar utama (pink sangat muda)
    C2  = "#fce4ec"   # sidebar
    C3  = "#f48fb1"   # aksen utama (pink medium)
    C4  = "#e91e8c"   # aksen tebal
    C5  = "#ffffff"   # putih
    C6  = "#880e4f"   # teks gelap
    C7  = "#fce4ec"   # card bg
    C8  = "#f06292"   # tombol hover
    ICON = "🎀"
else:
    C1  = "#e8f4fd"   # latar utama (biru sangat muda)
    C2  = "#dbeafe"   # sidebar
    C3  = "#60a5fa"   # aksen utama
    C4  = "#1d4ed8"   # aksen tebal
    C5  = "#ffffff"   # putih
    C6  = "#1e3a5f"   # teks gelap
    C7  = "#eff6ff"   # card bg
    C8  = "#3b82f6"   # tombol hover
    ICON = "🎓"

st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Nunito', sans-serif;
    color: {C6};
}}

/* ===================== BACKGROUND ===================== */
.stApp {{
    background-color: {C1};
}}

/* ===================== SIDEBAR ===================== */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C2} 0%, {C5} 100%);
    border-right: 2px solid {C3}40;
}}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {{
    color: {C6} !important;
    font-weight: 600;
}}

/* ===================== TOMBOL ===================== */
.stButton > button {{
    background: linear-gradient(135deg, {C3}, {C4}) !important;
    color: {C5} !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 4px 14px {C3}60 !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.3px !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px {C4}50 !important;
    background: linear-gradient(135deg, {C4}, {C8}) !important;
}}

/* ===================== SELECTBOX ===================== */
.stSelectbox > div > div {{
    background: {C5} !important;
    border: 2px solid {C3}80 !important;
    border-radius: 12px !important;
    color: {C6} !important;
    font-weight: 600 !important;
}}

.stSelectbox label {{
    color: {C6} !important;
    font-weight: 700 !important;
}}

/* ===================== FILE UPLOADER ===================== */
[data-testid="stFileUploader"] {{
    background: {C5} !important;
    border: 2px dashed {C3} !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}}

/* ===================== TABS ===================== */
.stTabs [data-baseweb="tab-list"] {{
    background: {C5};
    border-radius: 14px;
    padding: 6px;
    gap: 6px;
    box-shadow: 0 2px 12px {C3}30;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important;
    color: {C6} !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.4rem 1rem !important;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {C3}, {C4}) !important;
    color: {C5} !important;
    box-shadow: 0 3px 10px {C3}60 !important;
}}

/* ===================== DATAFRAME ===================== */
.stDataFrame {{
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid {C3}40 !important;
}}

/* ===================== SUCCESS / ERROR ===================== */
.stSuccess {{
    background: #f0fdf4 !important;
    border-left: 4px solid #22c55e !important;
    border-radius: 10px !important;
}}

.stError {{
    background: #fff1f2 !important;
    border-left: 4px solid #f43f5e !important;
    border-radius: 10px !important;
}}

.stWarning {{
    background: #fffbeb !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 10px !important;
}}

.stInfo {{
    background: {C2} !important;
    border-left: 4px solid {C3} !important;
    border-radius: 10px !important;
}}

/* ===================== RADIO ===================== */
.stRadio > div {{
    gap: 6px;
}}

.stRadio label {{
    background: {C5};
    border: 1.5px solid {C3}60;
    border-radius: 10px;
    padding: 0.4rem 0.8rem;
    cursor: pointer;
    color: {C6} !important;
    font-weight: 600 !important;
    transition: all 0.2s;
}}

.stRadio label:hover {{
    border-color: {C3};
    background: {C2};
}}

/* ===================== DOWNLOAD BUTTON ===================== */
.stDownloadButton > button {{
    background: {C5} !important;
    color: {C4} !important;
    border: 2px solid {C3} !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    transition: all 0.2s !important;
}}

.stDownloadButton > button:hover {{
    background: {C2} !important;
    transform: translateY(-1px) !important;
}}

/* ===================== CUSTOM KOMPONEN ===================== */

.header-box {{
    background: linear-gradient(135deg, {C5} 0%, {C2} 100%);
    border: 2px solid {C3}60;
    border-radius: 22px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    text-align: center;
    box-shadow: 0 6px 30px {C3}25;
}}

.header-box h1 {{
    font-size: 2.2rem;
    font-weight: 900;
    color: {C4};
    margin: 0;
    letter-spacing: -0.5px;
}}

.header-box p {{
    color: {C6}99;
    font-size: 1rem;
    margin-top: 0.4rem;
    font-weight: 600;
}}

.card {{
    background: {C5};
    border: 1.5px solid {C3}50;
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 3px 16px {C3}15;
}}

.card-title {{
    font-size: 1rem;
    font-weight: 800;
    color: {C4};
    border-bottom: 2px solid {C3}40;
    padding-bottom: 0.5rem;
    margin-bottom: 0.9rem;
}}

.metric-pill {{
    background: linear-gradient(135deg, {C2}, {C5});
    border: 2px solid {C3}60;
    border-radius: 16px;
    padding: 1rem 0.8rem;
    text-align: center;
    box-shadow: 0 3px 12px {C3}20;
}}

.metric-pill .val {{
    font-size: 2rem;
    font-weight: 900;
    color: {C4};
    line-height: 1;
}}

.metric-pill .lbl {{
    font-size: 0.8rem;
    font-weight: 700;
    color: {C6}80;
    margin-top: 0.3rem;
}}

.badge-ya {{
    display: inline-block;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    font-size: 1.3rem;
    font-weight: 900;
    padding: 0.7rem 2.2rem;
    border-radius: 50px;
    box-shadow: 0 6px 20px #22c55e50;
    letter-spacing: 0.5px;
}}

.badge-tidak {{
    display: inline-block;
    background: linear-gradient(135deg, #f43f5e, #dc2626);
    color: white;
    font-size: 1.3rem;
    font-weight: 900;
    padding: 0.7rem 2.2rem;
    border-radius: 50px;
    box-shadow: 0 6px 20px #f43f5e50;
    letter-spacing: 0.5px;
}}

.step-badge {{
    display: inline-block;
    background: linear-gradient(135deg, {C3}, {C4});
    color: white;
    font-size: 0.8rem;
    font-weight: 800;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    margin-right: 0.5rem;
}}

.tabel-header {{
    background: linear-gradient(90deg, {C2}, {C5});
    border-radius: 10px;
    padding: 0.5rem 0.8rem;
    font-weight: 800;
    color: {C4};
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}}

footer {{visibility: hidden;}}
#MainMenu {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if 'dataset'  not in st.session_state:
    st.session_state.dataset  = pd.DataFrame()
if 'nb_model' not in st.session_state:
    st.session_state.nb_model = None

# =========================================================
# NAIVE BAYES MANUAL
# =========================================================

class NaiveBayesManual:

    def __init__(self, alpha=1):
        self.alpha      = alpha
        self.prior      = {}
        self.likelihood = {}
        self.kelas      = []
        self.fitur      = []

    def fit(self, X, y):
        self.kelas = sorted(y.unique().tolist())
        self.fitur = list(X.columns)
        total      = len(y)
        count_kls  = Counter(y)
        self.prior = {k: count_kls[k] / total for k in self.kelas}

        for k in self.kelas:
            self.likelihood[k] = {}
            X_k = X[y == k]
            for f in self.fitur:
                nilai_unik = X[f].unique()
                count_f    = Counter(X_k[f])
                self.likelihood[k][f] = {
                    v: (count_f.get(v, 0) + self.alpha) / (len(X_k) + self.alpha * len(nilai_unik))
                    for v in nilai_unik
                }
        return self

    def predict_proba_log(self, row):
        hasil = {}
        for k in self.kelas:
            lp = math.log(self.prior[k])
            for f in self.fitur:
                lh = self.likelihood[k][f].get(row[f], self.alpha / (self.alpha * 10 + 1))
                lp += math.log(lh + 1e-9)
            hasil[k] = lp
        return hasil

    def predict_one(self, row):
        return max(self.predict_proba_log(row), key=self.predict_proba_log(row).get)

    def predict(self, X):
        return [max(self.predict_proba_log(X.iloc[i]),
                    key=self.predict_proba_log(X.iloc[i]).get)
                for i in range(len(X))]


# =========================================================
# FUNGSI PREPROCESSING OTOMATIS
# =========================================================

def auto_preprocess(df_raw):

    # Paksa semua nilai jadi string agar tidak ada float/NaN error
    df_raw = df_raw.fillna('').astype(str)

    KATA = ['PENDAPATAN', 'PEKERJAAN', 'TANGGUNGAN', 'STATUS', 'LABEL', 'NAMA', 'NO']

    header_row = None
    for i in range(min(10, len(df_raw))):
        baris      = df_raw.iloc[i].str.upper()
        nilai_valid = [v for v in baris.values if isinstance(v, str)]
        gabung     = ' '.join(nilai_valid)
        cocok      = sum(1 for k in KATA if k in gabung)
        if cocok >= 3:
            header_row = i
            break

    if header_row is not None:
        df_raw.columns = df_raw.iloc[header_row].str.strip()
        df_raw         = df_raw.iloc[header_row + 1:].copy()
        df_raw.reset_index(drop=True, inplace=True)
    else:
        df_raw.columns = df_raw.columns.astype(str).str.strip()

    # Hapus kolom kosong / nan
    df_raw = df_raw.loc[:, ~df_raw.columns.isin(['nan', 'None', '', 'NaN'])]
    df_raw = df_raw.loc[:, df_raw.columns.notna()]

    # Hapus baris yang seluruhnya string kosong
    df_raw = df_raw.loc[~(df_raw == '').all(axis=1)]
    df_raw.reset_index(drop=True, inplace=True)

    def cari(kata):
        for c in df_raw.columns:
            if kata.upper() in str(c).upper():
                return c
        return None

    def pend(x):
        x = str(x).lower().replace('.','').replace(',','').replace('rp','').strip()
        a = ''.join(filter(str.isdigit, x))
        if not a: return np.nan
        a = int(a)
        if 500000 <= a <= 1000000:   return 'Rendah'
        if 1000001 <= a <= 4000000:  return 'Sedang'
        if a > 4000000:              return 'Tinggi'
        return np.nan

    def pek(x):
        x = str(x).lower().strip()
        peta = {'petani':'Petani','pns':'PNS','polisi':'Polisi',
                'wiraswasta':'Wiraswasta','buruh':'Buruh',
                'pedagang':'Pedagang','nelayan':'Nelayan',
                'swasta':'Swasta','honorer':'Honorer',
                'tidak bekerja':'Tidak Bekerja'}
        for k,v in peta.items():
            if k in x: return v
        if x and x != 'nan': return x.title()
        return np.nan

    def tang(x):
        try:
            n = int(float(str(x).strip()))
            if 1<=n<=2: return 'Sedikit'
            if 3<=n<=4: return 'Sedang'
            if n>=5:    return 'Banyak'
            return np.nan
        except: return np.nan

    def rumah(x):
        x = str(x).lower().strip()
        if 'milik' in x or 'sendiri' in x: return 'Milik Sendiri'
        if 'kontrak' in x or 'sewa' in x:  return 'Kontrak/Sewa'
        return np.nan

    def label(x):
        x = str(x).lower().strip()
        if x in ['ya','1','layak','iya']:           return 'Ya'
        if x in ['tidak','0','tidak layak']:         return 'Tidak'
        return np.nan

    cp = cari('PENDAPATAN')
    ck = cari('PEKERJAAN')
    ct = cari('TANGGUNGAN')
    cr = cari('STATUS RUMAH') or cari('RUMAH')
    cl = cari('LABEL')

    peta = {}
    if cp: peta['PENDAPATAN ORANG TUA'] = df_raw[cp].apply(pend)
    if ck: peta['PEKERJAAN ORANG TUA']  = df_raw[ck].apply(pek)
    if ct: peta['JUMLAH TANGGUNGAN']    = df_raw[ct].apply(tang)
    if cr: peta['STATUS RUMAH']         = df_raw[cr].apply(rumah)
    if cl: peta['LABEL']                = df_raw[cl].apply(label)

    out = pd.DataFrame(peta)
    out.dropna(inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def latih_model(df):
    X = df.drop(columns=['LABEL'])
    y = df['LABEL']
    m = NaiveBayesManual(alpha=1)
    m.fit(X, y)
    return m


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(f"""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <span style='font-size:2.5rem;'>{ICON}</span>
        <div style='font-size:1.2rem; font-weight:900; color:{C4}; margin-top:0.3rem;'>Naive Bayes BSM</div>
        <div style='font-size:0.8rem; color:{C6}80; font-weight:600;'>Bantuan Siswa Miskin</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu = st.radio(
        "Navigasi",
        [
            f"{ICON} Upload Dataset",
            "🔬 Preprocessing",
            "🧮 Perhitungan Manual",
            "📊 Evaluasi Model",
            "🔍 Prediksi Data Baru",
            "📝 Data Uji",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if not st.session_state.dataset.empty:
        df_s = st.session_state.dataset
        total_s = len(df_s)
        vc_s    = df_s['LABEL'].value_counts() if 'LABEL' in df_s.columns else {}

        st.markdown(f"""
        <div class='card' style='padding:0.9rem 1rem;'>
            <div class='card-title'>📋 Info Dataset</div>
            <div style='font-size:0.9rem; color:{C6};'>
                <b>Total :</b> {total_s} data<br>
        """, unsafe_allow_html=True)

        for k, v in vc_s.items():
            warna = "#22c55e" if k == "Ya" else "#f43f5e"
            st.markdown(f"<span style='color:{warna}; font-weight:700;'>● {k} : {v}</span><br>", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-size:0.75rem; color:{C6}60; text-align:center; margin-top:1rem; font-weight:600;'>
        © 2025 · Naive Bayes BSM<br>Metode Klasifikasi Bayesian
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HEADER UTAMA
# =========================================================

st.markdown(f"""
<div class='header-box'>
    <h1>{ICON} Sistem Prediksi Penerima BSM</h1>
    <p>Metode Naive Bayes · Bantuan Siswa Miskin · Prediksi Kelayakan Siswa</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# =========================================================
# TAB : UPLOAD DATASET
# =========================================================
# =========================================================

if menu == f"{ICON} Upload Dataset":

    st.markdown("## 📂 Upload Dataset")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>📌 Petunjuk Upload</div>
            <div style='font-size:0.9rem; line-height:1.8; color:{C6};'>
            Unggah file <b>CSV</b> (pemisah titik koma <b>;</b>) atau <b>Excel (.xlsx)</b><br><br>
            Kolom yang wajib ada :<br>
            <span style='color:{C4};'>●</span> PENDAPATAN ORANG TUA<br>
            <span style='color:{C4};'>●</span> PEKERJAAN ORANG TUA<br>
            <span style='color:{C4};'>●</span> JUMLAH TANGGUNGAN<br>
            <span style='color:{C4};'>●</span> STATUS RUMAH<br>
            <span style='color:{C4};'>●</span> LABEL (Ya / Tidak)<br><br>
            Kolom lain (NAMA, NIK, NISN, dll) otomatis diabaikan.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>🎯 Format Label</div>
            <div style='font-size:0.9rem; color:{C6};'>
            <span style='color:#22c55e; font-weight:800;'>✅ Ya</span><br>
            Layak menerima BSM<br><br>
            <span style='color:#f43f5e; font-weight:800;'>❌ Tidak</span><br>
            Tidak layak menerima BSM
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>⚙️ Preprocessing Otomatis</div>
            <div style='font-size:0.85rem; color:{C6};'>
            Pendapatan → Rendah / Sedang / Tinggi<br>
            Tanggungan → Sedikit / Sedang / Banyak<br>
            Rumah → Milik Sendiri / Kontrak-Sewa
            </div>
        </div>
        """, unsafe_allow_html=True)

    file = st.file_uploader(
        "Pilih File Dataset",
        type=["csv", "xlsx"],
        help="Upload file CSV atau Excel"
    )

    if file:
        try:
            if file.name.endswith(".csv"):
                try:
                    df_raw = pd.read_csv(file, sep=';', header=0, dtype=str)
                    if len(df_raw.columns) <= 1:
                        file.seek(0)
                        df_raw = pd.read_csv(file, sep=',', header=0, dtype=str)
                except Exception:
                    file.seek(0)
                    df_raw = pd.read_csv(file, dtype=str)
            else:
                # Baca Excel tanpa header, semua paksa string
                df_raw = pd.read_excel(file, header=None, dtype=str)
                df_raw = df_raw.fillna('').astype(str)

            st.success(f"✅ File **{file.name}** berhasil dibaca — {len(df_raw)} baris ditemukan")

            with st.expander("👁 Lihat Data Mentah (10 baris pertama)"):
                st.dataframe(df_raw.head(10), use_container_width=True)

            df_bersih = auto_preprocess(df_raw.copy())

            if df_bersih.empty:
                st.error("❌ Kolom tidak ditemukan. Pastikan format file sesuai.")
            else:
                st.session_state.dataset  = df_bersih
                st.session_state.nb_model = latih_model(df_bersih)

                st.success(f"✅ Preprocessing selesai! **{len(df_bersih)} data** siap digunakan.")

                df_prev = df_bersih.copy()
                df_prev.insert(0, "No", range(1, len(df_prev)+1))

                st.markdown(f"### 📋 Preview Dataset Bersih")
                st.dataframe(df_prev, use_container_width=True, height=420)

        except Exception as e:
            st.error(f"❌ Error : {e}")


# =========================================================
# TAB : PREPROCESSING
# =========================================================

elif menu == "🔬 Preprocessing":

    st.markdown("## 🔬 Detail Preprocessing")

    if st.session_state.dataset.empty:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df = st.session_state.dataset

        df_num = df.copy()
        df_num.insert(0, "No", range(1, len(df_num)+1))
        st.markdown("### 📋 Dataset Setelah Preprocessing")
        st.dataframe(df_num, use_container_width=True, height=420)

        st.markdown("---")
        st.markdown("### 📊 Distribusi Setiap Kolom")

        cols = st.columns(len(df.columns))
        for i, col in enumerate(df.columns):
            with cols[i]:
                vc = df[col].value_counts()
                st.markdown(f"""
                <div class='card' style='padding:0.9rem;'>
                    <div class='card-title' style='font-size:0.85rem;'>{col}</div>
                """, unsafe_allow_html=True)
                for val, cnt in vc.items():
                    pct = cnt / len(df) * 100
                    warna = "#22c55e" if val == "Ya" else ("#f43f5e" if val == "Tidak" else C4)
                    st.markdown(f"""
                    <div style='margin-bottom:6px;'>
                        <span style='color:{warna}; font-weight:700; font-size:0.85rem;'>{val}</span>
                        <span style='color:{C6}80; font-size:0.8rem;'> : {cnt} ({pct:.1f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🗂 Panduan Kategori Normalisasi")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>💰 Pendapatan Orang Tua</div>
                <div style='font-size:0.88rem; color:{C6}; line-height:2;'>
                <b style='color:{C4};'>Rendah</b> : Rp 500rb – Rp 1jt<br>
                <b style='color:{C4};'>Sedang</b> : Rp 1jt – Rp 4jt<br>
                <b style='color:{C4};'>Tinggi</b> : > Rp 4jt
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>👨‍👩‍👧 Jumlah Tanggungan</div>
                <div style='font-size:0.88rem; color:{C6}; line-height:2;'>
                <b style='color:{C4};'>Sedikit</b> : 1 – 2 orang<br>
                <b style='color:{C4};'>Sedang</b>  : 3 – 4 orang<br>
                <b style='color:{C4};'>Banyak</b>  : ≥ 5 orang
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>🏠 Status Rumah & Label</div>
                <div style='font-size:0.88rem; color:{C6}; line-height:2;'>
                <b style='color:{C4};'>Milik Sendiri</b> : Rumah sendiri<br>
                <b style='color:{C4};'>Kontrak/Sewa</b>  : Sewa/kontrak<br>
                <b style='color:#22c55e;'>Label Ya</b> / <b style='color:#f43f5e;'>Tidak</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "📥 Download CSV",
                data=df.to_csv(index=False, sep=';').encode('utf-8'),
                file_name="hasil_preprocessing_bsm.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_b:
            buf = io.BytesIO(); df.to_excel(buf, index=False)
            st.download_button(
                "📥 Download Excel",
                data=buf.getvalue(),
                file_name="hasil_preprocessing_bsm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


# =========================================================
# TAB : PERHITUNGAN MANUAL
# =========================================================

elif menu == "🧮 Perhitungan Manual":

    st.markdown("## 🧮 Perhitungan Naive Bayes Manual")

    if st.session_state.dataset.empty:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df        = st.session_state.dataset
        TARGET    = 'LABEL'
        FITUR     = [c for c in df.columns if c != TARGET]
        kelas_list = sorted(df[TARGET].unique().tolist())
        total     = len(df)
        count_kls = Counter(df[TARGET])

        # ==================== PRIOR ====================
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'><span class='step-badge'>Langkah 1</span> Hitung Prior P(C)</div>
        """, unsafe_allow_html=True)

        cols = st.columns(len(kelas_list) + 1)
        with cols[0]:
            st.markdown(f"""
            <div class='metric-pill'>
                <div class='val'>{total}</div>
                <div class='lbl'>Total Data (N)</div>
            </div>
            """, unsafe_allow_html=True)

        for i, k in enumerate(kelas_list):
            p = count_kls[k] / total
            warna = "#22c55e" if k == "Ya" else "#f43f5e"
            with cols[i+1]:
                st.markdown(f"""
                <div class='metric-pill'>
                    <div class='val' style='color:{warna};'>{round(p,4)}</div>
                    <div class='lbl'>P({k}) = {count_kls[k]}/{total}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ==================== LIKELIHOOD ====================
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'><span class='step-badge'>Langkah 2</span> Hitung Likelihood P(xi | C)</div>
        """, unsafe_allow_html=True)

        st.info("Menggunakan **Laplace Smoothing** (alpha = 1) → mencegah probabilitas bernilai nol")

        for f in FITUR:
            st.markdown(f"#### 🔹 {f}")

            nilai_unik = sorted(df[f].unique().astype(str))
            V          = len(nilai_unik)

            rows = []
            for v in nilai_unik:
                row_d = {"Nilai": v}
                for k in kelas_list:
                    X_k   = df[df[TARGET] == k]
                    n_k   = len(X_k)
                    c     = Counter(X_k[f].astype(str)).get(v, 0)
                    lh    = (c + 1) / (n_k + V)
                    row_d[f"P(xi|{k}) — ({c}+1)/({n_k}+{V})"] = round(lh, 6)
                rows.append(row_d)

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ==================== POSTERIOR ====================
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'><span class='step-badge'>Langkah 3</span> Rumus Posterior</div>
        """, unsafe_allow_html=True)

        st.latex(r"P(C \mid X) \propto P(C) \times \prod_{i=1}^{n} P(x_i \mid C)")
        st.markdown(f"<div style='color:{C6}; font-weight:600;'>Kelas dengan nilai <b>log-posterior terbesar</b> = hasil prediksi.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📄 Detail Perhitungan Lengkap (Teks Terminal)"):
            teks = f"{'='*55}\nENTROPI & PRIOR\n{'='*55}\n\nTotal Data = {total}\n"
            for k in kelas_list:
                p = count_kls[k] / total
                teks += f"P({k}) = {count_kls[k]} / {total} = {round(p,6)}\n"
            teks += f"\n{'='*55}\nLIKELIHOOD P(xi | C) — LAPLACE SMOOTHING\n{'='*55}\n"
            for k in kelas_list:
                X_k = df[df[TARGET] == k]
                teks += f"\n--- KELAS : {k} (n={len(X_k)}) ---\n"
                for f in FITUR:
                    nilai_unik = sorted(df[f].unique().astype(str))
                    V  = len(nilai_unik)
                    teks += f"\n  [ {f} ]\n"
                    for v in nilai_unik:
                        c  = Counter(X_k[f].astype(str)).get(v, 0)
                        lh = (c + 1) / (len(X_k) + V)
                        teks += f"    P({v} | {k}) = ({c}+1)/({len(X_k)}+{V}) = {round(lh,6)}\n"
            st.code(teks, language="text")


# =========================================================
# TAB : EVALUASI MODEL
# =========================================================

elif menu == "📊 Evaluasi Model":

    st.markdown("## 📊 Evaluasi Model Naive Bayes")

    if st.session_state.dataset.empty or st.session_state.nb_model is None:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df    = st.session_state.dataset
        model = st.session_state.nb_model
        TARGET = 'LABEL'
        FITUR  = [c for c in df.columns if c != TARGET]

        X      = df[FITUR]
        y_true = df[TARGET].tolist()
        y_pred = model.predict(X)
        labels = sorted(set(y_true))

        akurasi = accuracy_score(y_true, y_pred)
        cm      = confusion_matrix(y_true, y_pred, labels=labels)
        benar   = sum(a == p for a, p in zip(y_true, y_pred))
        salah   = len(y_true) - benar

        # METRIK
        st.markdown("### 📈 Metrik Utama")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class='metric-pill'>
                <div class='val'>{round(akurasi*100,2)}%</div>
                <div class='lbl'>Akurasi</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-pill'>
                <div class='val' style='color:#22c55e;'>{benar}</div>
                <div class='lbl'>Prediksi Benar</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-pill'>
                <div class='val' style='color:#f43f5e;'>{salah}</div>
                <div class='lbl'>Prediksi Salah</div>
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class='metric-pill'>
                <div class='val'>{len(y_true)}</div>
                <div class='lbl'>Total Data</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        col_kiri, col_kanan = st.columns(2)

        # CONFUSION MATRIX
        with col_kiri:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>🗂 Confusion Matrix</div>
            """, unsafe_allow_html=True)

            cm_df = pd.DataFrame(
                cm,
                index=[f"Aktual : {l}" for l in labels],
                columns=[f"Prediksi : {l}" for l in labels]
            )
            st.dataframe(cm_df, use_container_width=True)

            if len(labels) == 2:
                tn, fp, fn, tp = cm.ravel()
                st.markdown(f"""
                <div style='font-size:0.88rem; margin-top:0.8rem; line-height:2.2; color:{C6};'>
                <span style='color:#22c55e; font-weight:800;'>● TP = {tp}</span>  Pred Ya & Aktual Ya<br>
                <span style='color:#22c55e; font-weight:800;'>● TN = {tn}</span>  Pred Tidak & Aktual Tidak<br>
                <span style='color:#f43f5e; font-weight:800;'>● FP = {fp}</span>  Pred Ya, Aktual Tidak<br>
                <span style='color:#f43f5e; font-weight:800;'>● FN = {fn}</span>  Pred Tidak, Aktual Ya
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # CLASSIFICATION REPORT
        with col_kanan:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>📋 Classification Report</div>
            """, unsafe_allow_html=True)

            rpt = pd.DataFrame(
                classification_report(y_true, y_pred, output_dict=True)
            ).transpose().round(4)
            st.dataframe(rpt, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔎 Detail Prediksi per Data")

        df_hasil = df.copy()
        df_hasil["Prediksi"] = y_pred
        df_hasil["✔ Status"] = df_hasil.apply(
            lambda r: "✅ Benar" if r[TARGET] == r["Prediksi"] else "❌ Salah", axis=1
        )
        df_hasil.insert(0, "No", range(1, len(df_hasil)+1))
        st.dataframe(df_hasil, use_container_width=True, height=420)


# =========================================================
# TAB : PREDIKSI DATA BARU
# =========================================================

elif menu == "🔍 Prediksi Data Baru":

    st.markdown("## 🔍 Prediksi Data Baru")

    if st.session_state.nb_model is None:
        st.warning("⚠️ Upload dataset terlebih dahulu agar model terlatih.")
    else:
        model  = st.session_state.nb_model
        df_ref = st.session_state.dataset

        def pilihan(kolom):
            return sorted(df_ref[kolom].unique().tolist()) if kolom in df_ref.columns else []

        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>📝 Masukkan Data Siswa</div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            pendapatan = st.selectbox("💰 Pendapatan Orang Tua",
                pilihan('PENDAPATAN ORANG TUA') or ["Rendah","Sedang","Tinggi"])
            tanggungan = st.selectbox("👨‍👩‍👧 Jumlah Tanggungan",
                pilihan('JUMLAH TANGGUNGAN') or ["Sedikit","Sedang","Banyak"])

        with col2:
            pekerjaan = st.selectbox("💼 Pekerjaan Orang Tua",
                pilihan('PEKERJAAN ORANG TUA') or ["Petani","PNS","Wiraswasta","Buruh"])
            rumah     = st.selectbox("🏠 Status Rumah",
                pilihan('STATUS RUMAH') or ["Milik Sendiri","Kontrak/Sewa"])

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button(f"🔍  Prediksi Sekarang", use_container_width=True):

            row = {
                "PENDAPATAN ORANG TUA" : pendapatan,
                "PEKERJAAN ORANG TUA"  : pekerjaan,
                "JUMLAH TANGGUNGAN"    : tanggungan,
                "STATUS RUMAH"         : rumah,
            }

            proba  = model.predict_proba_log(row)
            hasil  = max(proba, key=proba.get)

            st.markdown("---")
            st.markdown("### 🎯 Hasil Prediksi")

            if hasil == "Ya":
                st.markdown("""
                <div style='text-align:center; padding:1.2rem;'>
                    <span class='badge-ya'>✅ LAYAK MENERIMA BSM</span>
                </div>
                """, unsafe_allow_html=True)
                st.success("Berdasarkan data yang dimasukkan, siswa ini **LAYAK** menerima Bantuan Siswa Miskin.")
            else:
                st.markdown("""
                <div style='text-align:center; padding:1.2rem;'>
                    <span class='badge-tidak'>❌ TIDAK LAYAK MENERIMA BSM</span>
                </div>
                """, unsafe_allow_html=True)
                st.error("Berdasarkan data yang dimasukkan, siswa ini **TIDAK LAYAK** menerima Bantuan Siswa Miskin.")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"""
                <div class='card'>
                    <div class='card-title'>📋 Data Input</div>
                """, unsafe_allow_html=True)
                for k, v in row.items():
                    st.markdown(f"<div style='font-size:0.9rem; color:{C6}; padding:3px 0;'><b>{k}</b> : {v}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_b:
                st.markdown(f"""
                <div class='card'>
                    <div class='card-title'>🧮 Log Posterior per Kelas</div>
                """, unsafe_allow_html=True)
                for k, v in proba.items():
                    warna  = "#22c55e" if k == hasil else "#f43f5e"
                    terpil = " ← Terpilih" if k == hasil else ""
                    st.markdown(f"""
                    <div style='font-size:0.9rem; color:{warna}; font-weight:700; padding:3px 0;'>
                    {k} : {round(v,6)}{terpil}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB : DATA UJI
# =========================================================

elif menu == "📝 Data Uji":

    st.markdown("## 📝 Kelola Data Uji")

    if st.session_state.nb_model is None:
        st.warning("⚠️ Upload dataset utama terlebih dahulu.")
    else:
        model  = st.session_state.nb_model
        df_ref = st.session_state.dataset

        tab1, tab2 = st.tabs(["➕ Tambah Data Manual", "📂 Import Dataset Baru"])

        def pilihan2(kolom):
            return sorted(df_ref[kolom].unique().tolist()) if kolom in df_ref.columns else []

        with tab1:

            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>✏️ Input Data Baru</div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                t_pend = st.selectbox("💰 Pendapatan", pilihan2('PENDAPATAN ORANG TUA') or ["Rendah","Sedang","Tinggi"], key="dp1")
                t_tang = st.selectbox("👨‍👩‍👧 Tanggungan", pilihan2('JUMLAH TANGGUNGAN') or ["Sedikit","Sedang","Banyak"], key="dp2")
            with col2:
                t_pek  = st.selectbox("💼 Pekerjaan",  pilihan2('PEKERJAAN ORANG TUA') or ["Petani","PNS","Wiraswasta"], key="dp3")
                t_rmh  = st.selectbox("🏠 Status Rumah", pilihan2('STATUS RUMAH') or ["Milik Sendiri","Kontrak/Sewa"], key="dp4")

            st.markdown("</div>", unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if st.button("🔍 Prediksi & Tambah", use_container_width=True):
                    row = {"PENDAPATAN ORANG TUA":t_pend,"PEKERJAAN ORANG TUA":t_pek,
                           "JUMLAH TANGGUNGAN":t_tang,"STATUS RUMAH":t_rmh}
                    hasil = model.predict_one(row)
                    row["LABEL"] = hasil
                    st.session_state.dataset = pd.concat(
                        [pd.DataFrame([row]), st.session_state.dataset], ignore_index=True)
                    if hasil == "Ya":
                        st.success(f"✅ Ditambahkan — Prediksi : **{hasil}** (Layak BSM)")
                    else:
                        st.error(f"❌ Ditambahkan — Prediksi : **{hasil}** (Tidak Layak)")

            with col_b:
                if st.button("🗑 Hapus Data Terakhir", use_container_width=True):
                    if len(st.session_state.dataset) > 0:
                        st.session_state.dataset = st.session_state.dataset.iloc[:-1].reset_index(drop=True)
                        st.warning("Data terakhir berhasil dihapus.")

            with col_c:
                csv_dl = st.session_state.dataset.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button("💾 Simpan Dataset", data=csv_dl,
                    file_name="dataset_bsm_terbaru.csv", mime="text/csv", use_container_width=True)

            st.markdown("### 📋 Dataset Terkini")
            df_show = st.session_state.dataset.copy()
            df_show.insert(0, "No", range(1, len(df_show)+1))
            st.dataframe(df_show, use_container_width=True, height=420)

        with tab2:

            st.info("File baru akan digabung ke dataset yang sudah ada, lalu model dilatih ulang.")

            file_baru = st.file_uploader("Upload File CSV / Excel", type=["csv","xlsx"], key="fb")

            if file_baru and st.button("📂 Import & Gabung"):
                try:
                    if file_baru.name.endswith(".csv"):
                        df_imp = pd.read_csv(file_baru, sep=';', dtype=str)
                    else:
                        df_imp = pd.read_excel(file_baru, dtype=str)

                    df_clean = auto_preprocess(df_imp)

                    if df_clean.empty:
                        st.error("❌ Kolom tidak ditemukan.")
                    else:
                        st.session_state.dataset = pd.concat(
                            [st.session_state.dataset, df_clean], ignore_index=True)
                        st.session_state.nb_model = latih_model(st.session_state.dataset)
                        st.success(f"✅ {len(df_clean)} data ditambahkan. Model dilatih ulang.")
                except Exception as e:
                    st.error(f"❌ Error : {e}")
