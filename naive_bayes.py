# =========================================================
# APLIKASI NAIVE BAYES - KELAYAKAN PENERIMA BSM
# STREAMLIT - PYTHON
# =========================================================

# =========================================================
# INSTALL (jalankan di terminal sebelum run):
# pip install streamlit pandas numpy openpyxl scikit-learn
# Jalankan : streamlit run app.py
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import Counter, defaultdict
import io

# sklearn untuk evaluasi saja
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title  = "Naive Bayes - BSM",
    page_icon   = "🎓",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# =========================================================
# CSS CUSTOM
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #e2e8f0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid #334155;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a5f3fc;
}

/* Header utama */
.main-header {
    background: linear-gradient(135deg, #1e293b, #312e81);
    border: 1px solid #4f46e5;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(99,102,241,0.2);
}

.main-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #a5b4fc;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0.5rem;
}

/* Card */
.card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

.card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a5b4fc;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Badge hasil prediksi */
.badge-ya {
    display: inline-block;
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    font-size: 1.4rem;
    font-weight: 800;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    box-shadow: 0 0 20px rgba(16,185,129,0.4);
}

.badge-tidak {
    display: inline-block;
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white;
    font-size: 1.4rem;
    font-weight: 800;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    box-shadow: 0 0 20px rgba(239,68,68,0.4);
}

/* Metric box */
.metric-box {
    background: rgba(30,41,59,0.9);
    border: 1px solid #4f46e5;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #a5f3fc;
}

.metric-label {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.3rem;
}

/* Tabel warna */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(79,70,229,0.5) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #1e293b !important;
    border: 1px solid #4f46e5 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* Success/Error/Info */
.stSuccess {
    background: rgba(16,185,129,0.15) !important;
    border: 1px solid #10b981 !important;
    border-radius: 10px !important;
}

.stError {
    background: rgba(239,68,68,0.15) !important;
    border: 1px solid #ef4444 !important;
    border-radius: 10px !important;
}

/* Tab */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(30,41,59,0.5);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
}

/* Kode / mono */
.mono {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #a5f3fc;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if 'dataset' not in st.session_state:
    st.session_state.dataset = pd.DataFrame()

if 'nb_model' not in st.session_state:
    st.session_state.nb_model = None

# =========================================================
# FUNGSI NAIVE BAYES MANUAL
# =========================================================

class NaiveBayesManual:
    """
    Naive Bayes Kategorik Manual
    Menggunakan Laplace Smoothing (alpha=1)
    """

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.prior = {}
        self.likelihood = {}
        self.kelas = []
        self.fitur = []

    def fit(self, X, y):

        self.kelas = list(y.unique())
        self.fitur = list(X.columns)

        total = len(y)

        # === PRIOR P(C) ===
        count_kelas = Counter(y)
        self.prior = {
            k: count_kelas[k] / total
            for k in self.kelas
        }

        # === LIKELIHOOD P(xi | C) ===
        self.likelihood = {}

        for k in self.kelas:

            self.likelihood[k] = {}

            X_k = X[y == k]

            for f in self.fitur:

                nilai_unik = X[f].unique()
                count_f = Counter(X_k[f])

                self.likelihood[k][f] = {}

                for v in nilai_unik:

                    # Laplace Smoothing
                    self.likelihood[k][f][v] = (
                        (count_f.get(v, 0) + self.alpha)
                        / (len(X_k) + self.alpha * len(nilai_unik))
                    )

        return self

    def predict_proba(self, row):
        """
        Hitung posterior untuk satu baris (dict / Series)
        Return: dict {kelas: posterior}
        """

        hasil = {}

        for k in self.kelas:

            log_prob = math.log(self.prior[k])

            for f in self.fitur:

                val = row[f]

                lh = self.likelihood[k][f].get(val, self.alpha / (self.alpha * 10))

                log_prob += math.log(lh + 1e-9)

            hasil[k] = log_prob

        return hasil

    def predict_one(self, row):

        proba = self.predict_proba(row)
        return max(proba, key=proba.get)

    def predict(self, X):

        return [self.predict_one(X.iloc[i]) for i in range(len(X))]


# =========================================================
# FUNGSI PREPROCESSING OTOMATIS
# =========================================================

def kategori_pendapatan(x):
    x = str(x).lower().replace('.', '').replace(',', '').replace('rp', '').strip()
    angka = ''.join(filter(str.isdigit, x))
    if not angka:
        return np.nan
    angka = int(angka)
    if 500000 <= angka <= 1000000:
        return 'Rendah'
    elif 1000001 <= angka <= 4000000:
        return 'Sedang'
    elif angka > 4000000:
        return 'Tinggi'
    return np.nan

def kategori_pekerjaan(x):
    x = str(x).lower().strip()
    peta = {
        'petani': 'Petani', 'pns': 'PNS', 'polisi': 'Polisi',
        'wiraswasta': 'Wiraswasta', 'buruh': 'Buruh',
        'pedagang': 'Pedagang', 'nelayan': 'Nelayan',
        'swasta': 'Swasta', 'honorer': 'Honorer',
        'tidak bekerja': 'Tidak Bekerja',
    }
    for k, v in peta.items():
        if k in x:
            return v
    if x and x != 'nan':
        return x.title()
    return np.nan

def kategori_tanggungan(x):
    try:
        n = int(float(str(x).strip()))
        if 1 <= n <= 2:
            return 'Sedikit'
        elif 3 <= n <= 4:
            return 'Sedang'
        elif n >= 5:
            return 'Banyak'
        return np.nan
    except:
        return np.nan

def kategori_rumah(x):
    x = str(x).lower().strip()
    if 'milik' in x or 'sendiri' in x:
        return 'Milik Sendiri'
    elif 'kontrak' in x or 'sewa' in x:
        return 'Kontrak/Sewa'
    return np.nan

def kategori_label(x):
    x = str(x).lower().strip()
    if x in ['ya', '1', 'layak', 'iya']:
        return 'Ya'
    elif x in ['tidak', '0', 'tidak layak']:
        return 'Tidak'
    return np.nan

def preprocessing_otomatis(df_raw):
    """
    Auto-detect header & kolom, lalu normalisasi
    """
    # Cek apakah ada header ganda (seperti format colab)
    # Kalau baris 0 bukan nama kolom yang benar, cari header
    kolom_target = ['PENDAPATAN ORANG TUA', 'PEKERJAAN', 'JUMLAH TANGGUNGAN', 'STATUS RUMAH', 'LABEL']

    # Cek apakah header sudah langsung di baris 0
    header_ok = any(
        any(k in str(c).upper() for k in kolom_target)
        for c in df_raw.columns
    )

    if not header_ok:
        # Cari baris yang mengandung header
        for i in range(min(5, len(df_raw))):
            row = df_raw.iloc[i].astype(str).str.upper()
            if any(k in ' '.join(row.values) for k in kolom_target):
                df_raw.columns = df_raw.iloc[i]
                df_raw = df_raw[i+1:].copy()
                df_raw.reset_index(drop=True, inplace=True)
                break

    df_raw.columns = df_raw.columns.astype(str).str.strip()

    # Temukan kolom relevan
    def cari_kolom(kunci_list):
        for c in df_raw.columns:
            for k in kunci_list:
                if k.upper() in c.upper():
                    return c
        return None

    col_pendapatan  = cari_kolom(['PENDAPATAN'])
    col_pekerjaan   = cari_kolom(['PEKERJAAN'])
    col_tanggungan  = cari_kolom(['TANGGUNGAN'])
    col_rumah       = cari_kolom(['STATUS RUMAH', 'RUMAH'])
    col_label       = cari_kolom(['LABEL'])

    kolom_map = {
        col_pendapatan : ('PENDAPATAN ORANG TUA', kategori_pendapatan),
        col_pekerjaan  : ('PEKERJAAN ORANG TUA',  kategori_pekerjaan),
        col_tanggungan : ('JUMLAH TANGGUNGAN',     kategori_tanggungan),
        col_rumah      : ('STATUS RUMAH',           kategori_rumah),
        col_label      : ('LABEL',                  kategori_label),
    }

    hasil = {}

    for col_asli, (nama_baru, fungsi) in kolom_map.items():
        if col_asli:
            hasil[nama_baru] = df_raw[col_asli].apply(fungsi)

    df_out = pd.DataFrame(hasil)
    df_out.dropna(inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    return df_out


# =========================================================
# HITUNG MANUAL DETAIL (untuk tampilan)
# =========================================================

def hitung_manual_detail(df, target='LABEL'):

    fitur = [c for c in df.columns if c != target]
    kelas_list = df[target].unique().tolist()

    teks = ""

    total = len(df)
    count_kelas = Counter(df[target])

    teks += "=" * 55 + "\n"
    teks += "LANGKAH 1 : HITUNG PRIOR P(C)\n"
    teks += "=" * 55 + "\n"

    for k in kelas_list:
        p = count_kelas[k] / total
        teks += f"P({k}) = {count_kelas[k]} / {total} = {round(p, 6)}\n"

    teks += "\n" + "=" * 55 + "\n"
    teks += "LANGKAH 2 : HITUNG LIKELIHOOD P(xi | C)\n"
    teks += "=" * 55 + "\n"

    for k in kelas_list:
        X_k = df[df[target] == k]
        n_k = len(X_k)
        teks += f"\n--- KELAS : {k} (n={n_k}) ---\n"

        for f in fitur:
            nilai_unik = df[f].unique()
            count_f = Counter(X_k[f])
            teks += f"\n  [ {f} ]\n"

            for v in sorted(nilai_unik.astype(str)):
                c = count_f.get(v, 0)
                # Laplace smoothing
                lh = (c + 1) / (n_k + len(nilai_unik))
                teks += f"    P({v} | {k}) = ({c}+1) / ({n_k}+{len(nilai_unik)}) = {round(lh,6)}\n"

    teks += "\n" + "=" * 55 + "\n"
    teks += "LANGKAH 3 : POSTERIOR P(C | X)\n"
    teks += "=" * 55 + "\n"
    teks += "Untuk prediksi, hitung posterior setiap kelas\n"
    teks += "lalu pilih kelas dengan nilai terbesar.\n"
    teks += "P(C|X) ∝ P(C) × ∏ P(xi|C)\n"

    return teks


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🎓 Naive Bayes BSM")
    st.markdown("---")

    st.markdown("### 📌 Menu")
    menu = st.radio(
        "",
        [
            "📂 Upload Dataset",
            "🔬 Preprocessing",
            "🧮 Perhitungan Manual",
            "📊 Evaluasi Model",
            "🔍 Prediksi Data Baru",
            "📝 Data Uji"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if not st.session_state.dataset.empty:
        df_s = st.session_state.dataset
        st.markdown("### 📋 Info Dataset")
        st.markdown(f"**Total Data :** {len(df_s)}")

        if 'LABEL' in df_s.columns:
            vc = df_s['LABEL'].value_counts()
            for k, v in vc.items():
                st.markdown(f"**{k} :** {v}")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem;color:#64748b;'>
    Algoritma Naive Bayes<br>
    Prediksi Kelayakan BSM<br>
    © 2025
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HEADER UTAMA
# =========================================================

st.markdown("""
<div class="main-header">
    <h1>🎓 SISTEM PREDIKSI PENERIMA BSM</h1>
    <p>Metode Naive Bayes | Bantuan Siswa Miskin</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TAB : UPLOAD DATASET
# =========================================================

if menu == "📂 Upload Dataset":

    st.markdown("## 📂 Upload Dataset")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class='card'>
        <div class='card-title'>📌 Petunjuk Upload</div>
        File CSV (separator <b>;</b>) atau Excel (.xlsx)<br><br>
        Kolom yang wajib ada di dataset :<br>
        <ul>
        <li>PENDAPATAN ORANG TUA</li>
        <li>PEKERJAAN ORANG TUA</li>
        <li>JUMLAH TANGGUNGAN</li>
        <li>STATUS RUMAH</li>
        <li>LABEL</li>
        </ul>
        Kolom lain (NAMA, NIK, NISN, dll) akan otomatis diabaikan.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
        <div class='card-title'>📊 Format LABEL</div>
        <b>Ya</b> → Layak menerima BSM<br><br>
        <b>Tidak</b> → Tidak layak BSM
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
                df_raw = pd.read_csv(file, sep=';', header=0, dtype=str)
                # Coba juga tanpa header jika ada format khusus
                if len(df_raw.columns) == 1:
                    df_raw = pd.read_csv(file, sep=',', header=0, dtype=str)
            else:
                df_raw = pd.read_excel(file, header=0, dtype=str)

            st.success(f"✅ File berhasil dibaca : {file.name}")
            st.markdown(f"**Total baris :** {len(df_raw)} | **Total kolom :** {len(df_raw.columns)}")

            with st.expander("👁 Lihat Data Mentah"):
                st.dataframe(df_raw.head(20), use_container_width=True)

            # Preprocessing otomatis
            df_bersih = preprocessing_otomatis(df_raw.copy())

            if df_bersih.empty:
                st.error("❌ Kolom yang dibutuhkan tidak ditemukan. Cek format file Anda.")
            else:
                st.session_state.dataset = df_bersih

                # Latih model
                X = df_bersih.drop(columns=['LABEL'])
                y = df_bersih['LABEL']
                model = NaiveBayesManual(alpha=1)
                model.fit(X, y)
                st.session_state.nb_model = model

                st.success(f"✅ Preprocessing selesai! {len(df_bersih)} data siap digunakan.")

                # Preview bersih
                df_preview = df_bersih.copy()
                df_preview.insert(0, "No", range(1, len(df_preview)+1))
                st.markdown("### 📋 Preview Data Setelah Preprocessing")
                st.dataframe(df_preview, use_container_width=True, height=400)

        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")


# =========================================================
# TAB : PREPROCESSING
# =========================================================

elif menu == "🔬 Preprocessing":

    st.markdown("## 🔬 Detail Preprocessing")

    if st.session_state.dataset.empty:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu Upload Dataset.")
    else:
        df = st.session_state.dataset

        st.markdown("### 📋 Dataset Setelah Preprocessing")

        df_num = df.copy()
        df_num.insert(0, "No", range(1, len(df_num)+1))
        st.dataframe(df_num, use_container_width=True, height=400)

        st.markdown("---")
        st.markdown("### 📊 Distribusi Setiap Kolom")

        cols = st.columns(len(df.columns))

        for i, col in enumerate(df.columns):
            with cols[i]:
                vc = df[col].value_counts()
                st.markdown(f"**{col}**")
                for val, cnt in vc.items():
                    pct = cnt / len(df) * 100
                    st.markdown(f"`{val}` : **{cnt}** ({pct:.1f}%)")

        st.markdown("---")
        st.markdown("### 🗂 Kategori Normalisasi")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class='card'>
            <div class='card-title'>💰 Pendapatan Orang Tua</div>
            <b>Rendah</b> : Rp 500.000 – Rp 1.000.000<br>
            <b>Sedang</b> : Rp 1.000.001 – Rp 4.000.000<br>
            <b>Tinggi</b> : > Rp 4.000.000
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='card'>
            <div class='card-title'>👨‍👩‍👧 Jumlah Tanggungan</div>
            <b>Sedikit</b> : 1 – 2 orang<br>
            <b>Sedang</b>  : 3 – 4 orang<br>
            <b>Banyak</b>  : ≥ 5 orang
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='card'>
            <div class='card-title'>🏠 Status Rumah</div>
            <b>Milik Sendiri</b> : Rumah milik sendiri<br>
            <b>Kontrak/Sewa</b>  : Rumah kontrak atau sewa
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='card'>
            <div class='card-title'>🎯 Label</div>
            <b>Ya</b>    : Layak menerima BSM<br>
            <b>Tidak</b> : Tidak layak menerima BSM
            </div>
            """, unsafe_allow_html=True)

        # Download hasil preprocessing
        st.markdown("---")
        st.markdown("### 💾 Download Hasil Preprocessing")

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            csv_bytes = df.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                data=csv_bytes,
                file_name="hasil_preprocessing_bsm.csv",
                mime="text/csv"
            )

        with col_dl2:
            buf = io.BytesIO()
            df.to_excel(buf, index=False)
            st.download_button(
                "📥 Download Excel",
                data=buf.getvalue(),
                file_name="hasil_preprocessing_bsm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# =========================================================
# TAB : PERHITUNGAN MANUAL
# =========================================================

elif menu == "🧮 Perhitungan Manual":

    st.markdown("## 🧮 Perhitungan Naive Bayes Manual")

    if st.session_state.dataset.empty:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df = st.session_state.dataset
        target = 'LABEL'
        fitur = [c for c in df.columns if c != target]
        kelas_list = df[target].unique().tolist()

        # =====================================================
        # PRIOR
        # =====================================================
        st.markdown("### 📌 Langkah 1 : Hitung Prior P(C)")

        total = len(df)
        count_kelas = Counter(df[target])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{total}</div>
                <div class='metric-label'>Total Data</div>
            </div>
            """, unsafe_allow_html=True)

        for i, k in enumerate(kelas_list):
            c_idx = i + 2
            if c_idx <= 3:
                col = [col1, col2, col3][c_idx - 1]
                with col:
                    p = count_kelas[k] / total
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-value'>{round(p,4)}</div>
                        <div class='metric-label'>P({k}) = {count_kelas[k]}/{total}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # =====================================================
        # LIKELIHOOD
        # =====================================================
        st.markdown("### 📌 Langkah 2 : Hitung Likelihood P(xi | C)")
        st.info("Menggunakan **Laplace Smoothing** (alpha=1) untuk menghindari probabilitas nol.")

        for f in fitur:
            st.markdown(f"#### 🔹 Fitur : {f}")

            nilai_unik = sorted(df[f].unique().astype(str))

            tabel_data = {"Nilai": nilai_unik}

            for k in kelas_list:
                X_k = df[df[target] == k]
                n_k = len(X_k)
                count_f = Counter(X_k[f].astype(str))
                lh_list = []

                for v in nilai_unik:
                    c = count_f.get(v, 0)
                    lh = (c + 1) / (n_k + len(nilai_unik))
                    lh_list.append(
                        f"{c}+1 / {n_k}+{len(nilai_unik)} = {round(lh,6)}"
                    )

                tabel_data[f"P(xi | {k})"] = lh_list

            st.dataframe(
                pd.DataFrame(tabel_data),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")

        # =====================================================
        # DETAIL TEKS
        # =====================================================
        st.markdown("### 📌 Langkah 3 : Rumus Posterior")
        st.latex(r"P(C | X) \propto P(C) \times \prod_{i=1}^{n} P(x_i | C)")
        st.markdown("Kelas dengan nilai **posterior terbesar** = hasil prediksi.")

        with st.expander("📄 Lihat Detail Perhitungan Lengkap (Teks)"):
            teks = hitung_manual_detail(df)
            st.code(teks, language="text")


# =========================================================
# TAB : EVALUASI MODEL
# =========================================================

elif menu == "📊 Evaluasi Model":

    st.markdown("## 📊 Evaluasi Model Naive Bayes")

    if st.session_state.dataset.empty or st.session_state.nb_model is None:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df = st.session_state.dataset
        model = st.session_state.nb_model

        target = 'LABEL'
        fitur = [c for c in df.columns if c != target]

        X = df[fitur]
        y_true = df[target]

        y_pred = model.predict(X)

        labels = sorted(y_true.unique().tolist())

        # =====================================================
        # METRIK UTAMA
        # =====================================================
        akurasi = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        st.markdown("### 📈 Metrik Utama")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{round(akurasi*100,2)}%</div>
                <div class='metric-label'>Akurasi</div>
            </div>
            """, unsafe_allow_html=True)

        # TP TN FP FN
        if len(labels) == 2:
            tn, fp, fn, tp = cm.ravel()

            with col2:
                st.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{tp}</div>
                    <div class='metric-label'>True Positive (TP)</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{tn}</div>
                    <div class='metric-label'>True Negative (TN)</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{fp + fn}</div>
                    <div class='metric-label'>FP + FN (Salah)</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # =====================================================
        # CONFUSION MATRIX
        # =====================================================
        st.markdown("### 🗂 Confusion Matrix")

        cm_df = pd.DataFrame(
            cm,
            index=[f"Aktual : {l}" for l in labels],
            columns=[f"Prediksi : {l}" for l in labels]
        )

        st.dataframe(cm_df, use_container_width=True)

        # Keterangan TP TN FP FN
        if len(labels) == 2:
            ket = pd.DataFrame({
                "Keterangan":[
                    "TP (True Positive)",
                    "TN (True Negative)",
                    "FP (False Positive)",
                    "FN (False Negative)"
                ],
                "Jumlah": [tp, tn, fp, fn],
                "Penjelasan": [
                    "Prediksi Ya & Aktual Ya",
                    "Prediksi Tidak & Aktual Tidak",
                    "Prediksi Ya tapi Aktual Tidak",
                    "Prediksi Tidak tapi Aktual Ya"
                ]
            })
            st.dataframe(ket, use_container_width=True, hide_index=True)

        st.markdown("---")

        # =====================================================
        # CLASSIFICATION REPORT
        # =====================================================
        st.markdown("### 📋 Classification Report")

        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("---")

        # =====================================================
        # TABEL PREDIKSI VS AKTUAL
        # =====================================================
        st.markdown("### 🔎 Detail Prediksi per Data")

        df_hasil = df.copy()
        df_hasil["Prediksi"] = y_pred
        df_hasil["Status"] = df_hasil.apply(
            lambda r: "✅ Benar" if r[target] == r["Prediksi"] else "❌ Salah",
            axis=1
        )
        df_hasil.insert(0, "No", range(1, len(df_hasil)+1))

        st.dataframe(df_hasil, use_container_width=True, height=400)


# =========================================================
# TAB : PREDIKSI DATA BARU
# =========================================================

elif menu == "🔍 Prediksi Data Baru":

    st.markdown("## 🔍 Prediksi Data Baru")

    if st.session_state.nb_model is None:
        st.warning("⚠️ Upload dataset terlebih dahulu agar model terlatih.")
    else:
        model = st.session_state.nb_model

        st.markdown("### Masukkan Data Siswa")

        col1, col2 = st.columns(2)

        with col1:
            pendapatan = st.selectbox(
                "💰 Pendapatan Orang Tua",
                ["Rendah", "Sedang", "Tinggi"],
                help="Rendah: ≤1jt | Sedang: 1-4jt | Tinggi: >4jt"
            )

            tanggungan = st.selectbox(
                "👨‍👩‍👧 Jumlah Tanggungan",
                ["Sedikit", "Sedang", "Banyak"],
                help="Sedikit: 1-2 | Sedang: 3-4 | Banyak: ≥5"
            )

        with col2:
            pekerjaan = st.selectbox(
                "💼 Pekerjaan Orang Tua",
                sorted(model.likelihood.get(
                    list(model.kelas)[0], {}
                ).get('PEKERJAAN ORANG TUA', {}).keys())
                if model else
                ["Petani", "PNS", "Wiraswasta", "Buruh", "Pedagang", "Nelayan"]
            )

            rumah = st.selectbox(
                "🏠 Status Rumah",
                ["Milik Sendiri", "Kontrak/Sewa"]
            )

        if st.button("🔍 Prediksi Sekarang", use_container_width=True):

            row = {
                "PENDAPATAN ORANG TUA" : pendapatan,
                "PEKERJAAN ORANG TUA"  : pekerjaan,
                "JUMLAH TANGGUNGAN"    : tanggungan,
                "STATUS RUMAH"         : rumah
            }

            # Posterior
            proba = model.predict_proba(row)
            hasil = max(proba, key=proba.get)

            st.markdown("---")
            st.markdown("### 📊 Hasil Prediksi")

            if hasil == "Ya":
                st.markdown(f"""
                <div style='text-align:center; padding:1rem;'>
                    <span class='badge-ya'>✅ LAYAK MENERIMA BSM</span>
                </div>
                """, unsafe_allow_html=True)
                st.success("Siswa ini **LAYAK** menerima Bantuan Siswa Miskin (BSM).")
            else:
                st.markdown(f"""
                <div style='text-align:center; padding:1rem;'>
                    <span class='badge-tidak'>❌ TIDAK LAYAK MENERIMA BSM</span>
                </div>
                """, unsafe_allow_html=True)
                st.error("Siswa ini **TIDAK LAYAK** menerima Bantuan Siswa Miskin (BSM).")

            # Tabel input
            st.markdown("### 📋 Data yang Diinput")
            df_input = pd.DataFrame([row])
            st.dataframe(df_input, use_container_width=True, hide_index=True)

            # Detail probabilitas log
            st.markdown("### 🧮 Log Posterior per Kelas")

            prob_df = pd.DataFrame({
                "Kelas": list(proba.keys()),
                "Log Posterior": [round(v, 6) for v in proba.values()],
                "Prediksi": ["✅ Terpilih" if k == hasil else "" for k in proba.keys()]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)


# =========================================================
# TAB : DATA UJI
# =========================================================

elif menu == "📝 Data Uji":

    st.markdown("## 📝 Kelola & Uji Dataset")

    if st.session_state.nb_model is None:
        st.warning("⚠️ Upload dataset utama terlebih dahulu.")
    else:
        model = st.session_state.nb_model
        df_main = st.session_state.dataset

        tab1, tab2 = st.tabs(["➕ Tambah Data Manual", "📂 Import Dataset Baru"])

        # ==================================================
        # TAMBAH MANUAL
        # ==================================================
        with tab1:

            st.markdown("### Input Data Baru")

            col1, col2 = st.columns(2)

            with col1:
                t_pendapatan = st.selectbox(
                    "💰 Pendapatan Orang Tua",
                    ["Rendah", "Sedang", "Tinggi"],
                    key="t_pend"
                )
                t_tanggungan = st.selectbox(
                    "👨‍👩‍👧 Jumlah Tanggungan",
                    ["Sedikit", "Sedang", "Banyak"],
                    key="t_tang"
                )

            with col2:
                pekerjaan_choices = sorted(
                    model.likelihood.get(
                        list(model.kelas)[0], {}
                    ).get('PEKERJAAN ORANG TUA', {}).keys()
                ) if model else ["Petani", "PNS", "Wiraswasta", "Buruh"]

                t_pekerjaan = st.selectbox(
                    "💼 Pekerjaan Orang Tua",
                    pekerjaan_choices,
                    key="t_pek"
                )
                t_rumah = st.selectbox(
                    "🏠 Status Rumah",
                    ["Milik Sendiri", "Kontrak/Sewa"],
                    key="t_rumah"
                )

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if st.button("🔍 Prediksi & Tambah", use_container_width=True):
                    row = {
                        "PENDAPATAN ORANG TUA" : t_pendapatan,
                        "PEKERJAAN ORANG TUA"  : t_pekerjaan,
                        "JUMLAH TANGGUNGAN"    : t_tanggungan,
                        "STATUS RUMAH"         : t_rumah
                    }
                    hasil = model.predict_one(row)
                    row["LABEL"] = hasil

                    df_baru = pd.DataFrame([row])
                    st.session_state.dataset = pd.concat(
                        [df_baru, st.session_state.dataset],
                        ignore_index=True
                    )

                    if hasil == "Ya":
                        st.success(f"✅ Ditambahkan! Prediksi: **{hasil}** (Layak BSM)")
                    else:
                        st.error(f"❌ Ditambahkan! Prediksi: **{hasil}** (Tidak Layak BSM)")

            with col_b:
                if st.button("🗑 Hapus Data Terakhir", use_container_width=True):
                    if len(st.session_state.dataset) > 0:
                        st.session_state.dataset = st.session_state.dataset.iloc[:-1]
                        st.warning("Data terakhir dihapus.")

            with col_c:
                df_dl = st.session_state.dataset
                csv_dl = df_dl.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    "💾 Simpan Dataset",
                    data=csv_dl,
                    file_name="dataset_bsm_terbaru.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # Tampilkan dataset terkini
            st.markdown("### 📋 Dataset Terkini")
            df_show = st.session_state.dataset.copy()
            df_show.insert(0, "No", range(1, len(df_show)+1))
            st.dataframe(df_show, use_container_width=True, height=400)

        # ==================================================
        # IMPORT DATASET BARU
        # ==================================================
        with tab2:

            st.markdown("### Import Dataset Tambahan")
            st.info("File akan digabung dengan dataset yang sudah ada.")

            file_baru = st.file_uploader(
                "Upload File CSV / Excel",
                type=["csv", "xlsx"],
                key="file_baru"
            )

            if file_baru and st.button("📂 Import & Gabung"):
                try:
                    if file_baru.name.endswith(".csv"):
                        df_import = pd.read_csv(file_baru, sep=';', dtype=str)
                    else:
                        df_import = pd.read_excel(file_baru, dtype=str)

                    df_bersih = preprocessing_otomatis(df_import)

                    if df_bersih.empty:
                        st.error("❌ Kolom tidak ditemukan. Cek format file.")
                    else:
                        st.session_state.dataset = pd.concat(
                            [st.session_state.dataset, df_bersih],
                            ignore_index=True
                        )

                        # Latih ulang model
                        X = st.session_state.dataset.drop(columns=['LABEL'])
                        y = st.session_state.dataset['LABEL']
                        model_baru = NaiveBayesManual(alpha=1)
                        model_baru.fit(X, y)
                        st.session_state.nb_model = model_baru

                        st.success(f"✅ {len(df_bersih)} data berhasil ditambahkan. Model dilatih ulang.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
