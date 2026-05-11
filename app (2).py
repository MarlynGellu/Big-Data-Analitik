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
footer{{visibility:hidden;}}
#MainMenu{{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

for key, val in {
    'dataset'     : pd.DataFrame(),
    'nb_model'    : None,
    'data_uji'    : pd.DataFrame(),
    'hasil_uji'   : pd.DataFrame(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================================================
# NAIVE BAYES MANUAL
# =========================================================

class NaiveBayesManual:
    def __init__(self, alpha=1):
        self.alpha=alpha; self.prior={}; self.likelihood={}
        self.kelas=[]; self.fitur=[]

    def fit(self, X, y):
        self.kelas=sorted(y.unique().tolist()); self.fitur=list(X.columns)
        total=len(y); ck=Counter(y)
        self.prior={k: ck[k]/total for k in self.kelas}
        for k in self.kelas:
            self.likelihood[k]={}
            Xk=X[y==k]
            for f in self.fitur:
                nu=X[f].unique(); cf=Counter(Xk[f])
                self.likelihood[k][f]={
                    v:(cf.get(v,0)+self.alpha)/(len(Xk)+self.alpha*len(nu))
                    for v in nu}
        return self

    def predict_proba_log(self, row):
        h={}
        for k in self.kelas:
            lp=math.log(self.prior[k])
            for f in self.fitur:
                lh=self.likelihood[k][f].get(row[f], self.alpha/(self.alpha*10+1))
                lp+=math.log(lh+1e-9)
            h[k]=lp
        return h

    def predict_one(self, row):
        p=self.predict_proba_log(row)
        return max(p, key=p.get)

    def predict(self, X):
        return [self.predict_one(X.iloc[i]) for i in range(len(X))]

# =========================================================
# PREPROCESSING
# =========================================================

def auto_preprocess(df_raw):
    df_raw = df_raw.fillna('').astype(str)
    KATA = ['PENDAPATAN','PEKERJAAN','TANGGUNGAN','STATUS','LABEL','NAMA','NO']
    header_row=None
    for i in range(min(10,len(df_raw))):
        baris=df_raw.iloc[i].str.upper()
        gabung=' '.join([v for v in baris.values if isinstance(v,str)])
        if sum(1 for k in KATA if k in gabung)>=3:
            header_row=i; break
    if header_row is not None:
        df_raw.columns=df_raw.iloc[header_row].str.strip()
        df_raw=df_raw.iloc[header_row+1:].copy()
        df_raw.reset_index(drop=True,inplace=True)
    else:
        df_raw.columns=df_raw.columns.astype(str).str.strip()
    df_raw=df_raw.loc[:,~df_raw.columns.isin(['nan','None','','NaN'])]
    df_raw=df_raw.loc[:,df_raw.columns.notna()]
    df_raw=df_raw.loc[~(df_raw=='').all(axis=1)]
    df_raw.reset_index(drop=True,inplace=True)

    def cari(kata):
        for c in df_raw.columns:
            if kata.upper() in str(c).upper(): return c
        return None

    def pend(x):
        x=str(x).lower().replace('.','').replace(',','').replace('rp','').strip()
        a=''.join(filter(str.isdigit,x))
        if not a: return np.nan
        a=int(a)
        if 500000<=a<=1000000: return 'Rendah'
        if 1000001<=a<=4000000: return 'Sedang'
        if a>4000000: return 'Tinggi'
        return np.nan

    def pek(x):
        x=str(x).lower().strip()
        peta={'petani':'Petani','pns':'PNS','polisi':'Polisi',
              'wiraswasta':'Wiraswasta','buruh':'Buruh','pedagang':'Pedagang',
              'nelayan':'Nelayan','swasta':'Swasta','honorer':'Honorer',
              'tidak bekerja':'Tidak Bekerja'}
        for k,v in peta.items():
            if k in x: return v
        if x and x!='nan': return x.title()
        return np.nan

    def tang(x):
        try:
            n=int(float(str(x).strip()))
            if 1<=n<=2: return 'Sedikit'
            if 3<=n<=4: return 'Sedang'
            if n>=5: return 'Banyak'
            return np.nan
        except: return np.nan

    def rumah(x):
        x=str(x).lower().strip()
        if 'milik' in x or 'sendiri' in x: return 'Milik Sendiri'
        if 'kontrak' in x or 'sewa' in x: return 'Kontrak/Sewa'
        return np.nan

    def lbl(x):
        x=str(x).lower().strip()
        if x in ['ya','1','layak','iya']: return 'Ya'
        if x in ['tidak','0','tidak layak']: return 'Tidak'
        return np.nan

    cp=cari('PENDAPATAN'); ck=cari('PEKERJAAN')
    ct=cari('TANGGUNGAN'); cr=cari('STATUS RUMAH') or cari('RUMAH')
    cl=cari('LABEL')

    peta={}
    if cp: peta['PENDAPATAN ORANG TUA']=df_raw[cp].apply(pend)
    if ck: peta['PEKERJAAN ORANG TUA'] =df_raw[ck].apply(pek)
    if ct: peta['JUMLAH TANGGUNGAN']   =df_raw[ct].apply(tang)
    if cr: peta['STATUS RUMAH']        =df_raw[cr].apply(rumah)
    if cl: peta['LABEL']               =df_raw[cl].apply(lbl)

    out=pd.DataFrame(peta)
    out.dropna(inplace=True)
    out.reset_index(drop=True,inplace=True)
    return out

def latih_model(df):
    X=df.drop(columns=['LABEL']); y=df['LABEL']
    m=NaiveBayesManual(alpha=1); m.fit(X,y); return m

def baca_file(file):
    if file.name.endswith(".csv"):
        try:
            df=pd.read_csv(file,sep=';',header=0,dtype=str)
            if len(df.columns)<=1:
                file.seek(0); df=pd.read_csv(file,sep=',',header=0,dtype=str)
        except:
            file.seek(0); df=pd.read_csv(file,dtype=str)
    else:
        df=pd.read_excel(file,header=None,dtype=str)
        df=df.fillna('').astype(str)
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

    menu = st.radio("", [
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
            w="#22c55e" if k=="Ya" else "#f43f5e"
            st.markdown(f"<div style='color:{w};font-weight:700;font-size:.85rem;'>● {k} : {v}</div>", unsafe_allow_html=True)
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

    # Info sistem
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
            &nbsp;&nbsp;Rendah / Sedang / Tinggi<br>
            <span style='color:{C4};font-weight:800;'>② Pekerjaan Orang Tua</span><br>
            &nbsp;&nbsp;Petani / PNS / dll<br>
            <span style='color:{C4};font-weight:800;'>③ Jumlah Tanggungan</span><br>
            &nbsp;&nbsp;Sedikit / Sedang / Banyak<br>
            <span style='color:{C4};font-weight:800;'>④ Status Rumah</span><br>
            &nbsp;&nbsp;Milik Sendiri / Kontrak/Sewa<br>
            <span style='color:#22c55e;font-weight:800;'>⑤ Label (Target)</span><br>
            &nbsp;&nbsp;Ya (Layak) / Tidak (Tidak Layak)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Langkah penggunaan
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

    # Statistik dataset jika sudah upload
    if not st.session_state.dataset.empty:
        df=st.session_state.dataset
        st.markdown("### 📊 Statistik Dataset Aktif")
        c1,c2,c3,c4 = st.columns(4)
        total=len(df); vc=df['LABEL'].value_counts()
        ya=vc.get('Ya',0); tidak=vc.get('Tidak',0)
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
                st.dataframe(df_raw.head(10), use_container_width=True)

            with st.spinner("⏳ Preprocessing & Training model..."):
                df_bersih = auto_preprocess(df_raw.copy())

            if df_bersih.empty:
                st.error("❌ Kolom tidak ditemukan. Pastikan format file sesuai.")
            else:
                st.session_state.dataset  = df_bersih
                st.session_state.nb_model = latih_model(df_bersih)
                st.success(f"✅ Selesai! **{len(df_bersih)} data** berhasil diproses & model terlatih.")

                # Statistik hasil training
                st.markdown("### 📊 Hasil Training")
                vc=df_bersih['LABEL'].value_counts()
                c1,c2,c3,c4=st.columns(4)
                model=st.session_state.nb_model
                yp=model.predict(df_bersih.drop(columns=['LABEL']))
                acc=accuracy_score(df_bersih['LABEL'].tolist(), yp)

                for col, val, lbl_s, warna in [
                    (c1, len(df_bersih), "Total Data Training", C4),
                    (c2, vc.get('Ya',0), "Label : Ya", "#22c55e"),
                    (c3, vc.get('Tidak',0), "Label : Tidak", "#f43f5e"),
                    (c4, f"{round(acc*100,2)}%", "Akurasi Training", C4),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class='metric-pill'>
                            <div class='val' style='color:{warna};'>{val}</div>
                            <div class='lbl'>{lbl_s}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Pie distribusi kelas
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(vc.index), values=list(vc.values),
                    hole=.5, marker_colors=["#22c55e","#f43f5e"],
                    textinfo="label+percent", textfont_size=13,
                )])
                fig_pie.update_layout(
                    title=dict(text="Distribusi Label Dataset Training", font=dict(size=14)),
                    height=280, showlegend=False, **PLOT_LAYOUT)
                st.plotly_chart(fig_pie, use_container_width=True)

                # Preview dataset bersih
                st.markdown("### 📋 Preview Dataset Setelah Preprocessing")
                df_prev=df_bersih.copy()
                df_prev.insert(0,"No",range(1,len(df_prev)+1))
                st.dataframe(df_prev, use_container_width=True, height=400)

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
        df=st.session_state.dataset

        df_num=df.copy(); df_num.insert(0,"No",range(1,len(df_num)+1))
        st.markdown("### 📋 Dataset Setelah Preprocessing")
        st.dataframe(df_num, use_container_width=True, height=420)

        st.markdown("---")
        st.markdown("### 📊 Distribusi Setiap Kolom")

        cols=st.columns(len(df.columns))
        for i, col in enumerate(df.columns):
            with cols[i]:
                vc=df[col].value_counts()
                st.markdown(f"<div class='card' style='padding:.9rem;'><div class='card-title' style='font-size:.85rem;'>{col}</div>", unsafe_allow_html=True)
                for val, cnt in vc.items():
                    pct=cnt/len(df)*100
                    w="#22c55e" if val=="Ya" else ("#f43f5e" if val=="Tidak" else C4)
                    bar_w=int(pct/2)
                    st.markdown(f"""
                    <div style='margin-bottom:6px;'>
                        <div style='color:{w};font-weight:700;font-size:.85rem;'>{val}</div>
                        <div style='font-size:.78rem;color:{C6}80;'>{cnt} data ({pct:.1f}%)</div>
                        <div style='height:6px;background:{C2};border-radius:3px;margin-top:3px;'>
                            <div style='height:6px;width:{min(bar_w*2,100)}%;background:{w};border-radius:3px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🗂 Panduan Kategori Normalisasi")
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>💰 Pendapatan Orang Tua</div>
                <div style='font-size:.88rem;color:{C6};line-height:2;'>
                <b style='color:{C4};'>Rendah</b> : Rp 500rb – Rp 1jt<br>
                <b style='color:{C4};'>Sedang</b> : Rp 1jt – Rp 4jt<br>
                <b style='color:{C4};'>Tinggi</b> : &gt; Rp 4jt
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>👨‍👩‍👧 Jumlah Tanggungan</div>
                <div style='font-size:.88rem;color:{C6};line-height:2;'>
                <b style='color:{C4};'>Sedikit</b> : 1 – 2 orang<br>
                <b style='color:{C4};'>Sedang</b>  : 3 – 4 orang<br>
                <b style='color:{C4};'>Banyak</b>  : ≥ 5 orang
                </div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>🏠 Status Rumah & Label</div>
                <div style='font-size:.88rem;color:{C6};line-height:2;'>
                <b style='color:{C4};'>Milik Sendiri</b><br>
                <b style='color:{C4};'>Kontrak/Sewa</b><br>
                <b style='color:#22c55e;'>Label Ya</b> /
                <b style='color:#f43f5e;'>Tidak</b>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        c_a,c_b=st.columns(2)
        with c_a:
            st.download_button("📥 Download CSV", data=df_to_csv_bytes(df),
                file_name="preprocessing_bsm.csv", mime="text/csv", use_container_width=True)
        with c_b:
            st.download_button("📥 Download Excel", data=df_to_excel_bytes(df),
                file_name="preprocessing_bsm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)

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

        # PRIOR
        st.markdown(f"<div class='card'><div class='card-title'><span class='step-badge'>Langkah 1</span> Hitung Prior P(C)</div>", unsafe_allow_html=True)
        cols=st.columns(len(kelas_list)+1)
        with cols[0]:
            st.markdown(f"<div class='metric-pill'><div class='val'>{total}</div><div class='lbl'>Total Data (N)</div></div>", unsafe_allow_html=True)
        for i,k in enumerate(kelas_list):
            p=ck[k]/total; w="#22c55e" if k=="Ya" else "#f43f5e"
            with cols[i+1]:
                st.markdown(f"<div class='metric-pill'><div class='val' style='color:{w};'>{round(p,4)}</div><div class='lbl'>P({k}) = {ck[k]}/{total}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # LIKELIHOOD
        st.markdown(f"<div class='card'><div class='card-title'><span class='step-badge'>Langkah 2</span> Hitung Likelihood P(xi | C) dengan Laplace Smoothing</div>", unsafe_allow_html=True)
        st.info("**Laplace Smoothing** (alpha=1) digunakan agar tidak ada probabilitas bernilai nol.")

        for f in FITUR:
            st.markdown(f"#### 🔹 {f}")
            nilai_unik=sorted(df[f].unique().astype(str)); V=len(nilai_unik)
            rows=[]
            for v in nilai_unik:
                row_d={"Nilai":v}
                for k in kelas_list:
                    Xk=df[df[TARGET]==k]; nk=len(Xk)
                    c=Counter(Xk[f].astype(str)).get(v,0)
                    lh=(c+1)/(nk+V)
                    row_d[f"P(xi|{k}) — ({c}+1)/({nk}+{V})"]=round(lh,6)
                rows.append(row_d)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # POSTERIOR
        st.markdown(f"<div class='card'><div class='card-title'><span class='step-badge'>Langkah 3</span> Rumus Posterior & Prediksi</div>", unsafe_allow_html=True)
        st.latex(r"P(C \mid X) \propto P(C) \times \prod_{i=1}^{n} P(x_i \mid C)")
        st.markdown(f"<div style='color:{C6};font-weight:600;font-size:.9rem;'>Kelas dengan <b>log-posterior terbesar</b> dipilih sebagai hasil prediksi.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Detail teks
        with st.expander("📄 Lihat Perhitungan Lengkap (Format Terminal)"):
            teks=f"{'='*55}\nPRIOR P(C)\n{'='*55}\nTotal Data = {total}\n"
            for k in kelas_list:
                teks+=f"P({k}) = {ck[k]}/{total} = {round(ck[k]/total,6)}\n"
            teks+=f"\n{'='*55}\nLIKELIHOOD P(xi|C) — LAPLACE SMOOTHING\n{'='*55}\n"
            for k in kelas_list:
                Xk=df[df[TARGET]==k]
                teks+=f"\n--- KELAS : {k} (n={len(Xk)}) ---\n"
                for f in FITUR:
                    nu=sorted(df[f].unique().astype(str)); V=len(nu)
                    teks+=f"\n  [ {f} ]\n"
                    for v in nu:
                        c=Counter(Xk[f].astype(str)).get(v,0)
                        lh=(c+1)/(len(Xk)+V)
                        teks+=f"    P({v}|{k}) = ({c}+1)/({len(Xk)}+{V}) = {round(lh,6)}\n"
            st.code(teks, language="text")

# =========================================================
# 📊 EVALUASI & GRAFIK
# =========================================================

elif menu == "📊 Evaluasi & Grafik":

    st.markdown("## 📊 Evaluasi Model Naive Bayes")

    if st.session_state.dataset.empty or st.session_state.nb_model is None:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df=st.session_state.dataset; model=st.session_state.nb_model
        TARGET='LABEL'; FITUR=[c for c in df.columns if c!=TARGET]
        X=df[FITUR]; y_true=df[TARGET].tolist()
        y_pred=model.predict(X)
        labels=sorted(set(y_true))
        acc=accuracy_score(y_true,y_pred)
        cm=confusion_matrix(y_true,y_pred,labels=labels)
        benar=sum(a==p for a,p in zip(y_true,y_pred)); salah=len(y_true)-benar

        # METRIK
        st.markdown("### 📈 Metrik Utama")
        c1,c2,c3,c4=st.columns(4)
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
            fig_cm.update_layout(title=dict(text="Confusion Matrix Heatmap",font=dict(size=15)),
                height=350, **PLOT_LAYOUT)
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_kanan:
            if len(labels)==2:
                tn,fp,fn,tp=cm.ravel()
                fig_bar=go.Figure(data=[go.Bar(
                    x=["TP","TN","FP","FN"], y=[tp,tn,fp,fn],
                    text=[tp,tn,fp,fn], textposition="outside",
                    marker_color=["#22c55e","#22c55e","#f43f5e","#f43f5e"],
                    marker_line_width=0,
                )])
                fig_bar.update_layout(title=dict(text="TP / TN / FP / FN",font=dict(size=15)),
                    height=350, yaxis=dict(gridcolor="#e5e7eb"), showlegend=False, **PLOT_LAYOUT)
                st.plotly_chart(fig_bar, use_container_width=True)

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

        ca,cb,cc=st.columns(3)

        with ca:
            fig_pie=go.Figure(data=[go.Pie(
                labels=["Benar","Salah"], values=[benar,salah], hole=.5,
                marker_colors=["#22c55e","#f43f5e"],
                textinfo="label+percent", textfont_size=13,
            )])
            fig_pie.update_layout(title=dict(text="Akurasi Prediksi",font=dict(size=14)),
                height=300, showlegend=False, **PLOT_LAYOUT)
            st.plotly_chart(fig_pie, use_container_width=True)

        with cb:
            ck_eval=Counter(y_true)
            fig_dist=go.Figure(data=[go.Bar(
                x=list(ck_eval.keys()), y=list(ck_eval.values()),
                text=list(ck_eval.values()), textposition="outside",
                marker_color=["#22c55e" if k=="Ya" else "#f43f5e" for k in ck_eval],
            )])
            fig_dist.update_layout(title=dict(text="Distribusi Kelas Aktual",font=dict(size=14)),
                height=300, yaxis=dict(gridcolor="#e5e7eb"), showlegend=False, **PLOT_LAYOUT)
            st.plotly_chart(fig_dist, use_container_width=True)

        with cc:
            ck_pred=Counter(y_pred)
            fig_pr=go.Figure(data=[go.Bar(
                x=list(ck_pred.keys()), y=list(ck_pred.values()),
                text=list(ck_pred.values()), textposition="outside",
                marker_color=[C3]*len(ck_pred),
            )])
            fig_pr.update_layout(title=dict(text="Distribusi Kelas Prediksi",font=dict(size=14)),
                height=300, yaxis=dict(gridcolor="#e5e7eb"), showlegend=False, **PLOT_LAYOUT)
            st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Classification Report")

        c_l2, c_r2 = st.columns(2)
        with c_l2:
            rpt=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True)).transpose().round(4)
            st.dataframe(rpt, use_container_width=True)

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
            fig_rpt.update_layout(
                barmode="group",
                title=dict(text="Precision / Recall / F1-Score",font=dict(size=14)),
                height=320, yaxis=dict(gridcolor="#e5e7eb",range=[0,1.2]),
                legend=dict(orientation="h",y=1.18), **PLOT_LAYOUT)
            st.plotly_chart(fig_rpt, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔎 Detail Prediksi per Data")
        df_hasil=df.copy()
        df_hasil["Prediksi"]=y_pred
        df_hasil["Status"]=["✅ Benar" if a==p else "❌ Salah" for a,p in zip(y_true,y_pred)]
        df_hasil.insert(0,"No",range(1,len(df_hasil)+1))
        st.dataframe(df_hasil, use_container_width=True, height=400)

        st.markdown("---")
        c_d1,c_d2=st.columns(2)
        with c_d1:
            st.download_button("📥 Download Hasil Prediksi CSV",
                data=df_to_csv_bytes(df_hasil), file_name="hasil_prediksi_training.csv",
                mime="text/csv", use_container_width=True)
        with c_d2:
            st.download_button("📥 Download Hasil Prediksi Excel",
                data=df_to_excel_bytes(df_hasil), file_name="hasil_prediksi_training.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)

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

        def pil(kolom):
            return sorted(df_ref[kolom].unique().tolist()) if kolom in df_ref.columns else []

        st.markdown(f"<div class='card'><div class='card-title'>✏️ Masukkan Data Siswa</div>", unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            pendapatan=st.selectbox("💰 Pendapatan Orang Tua", pil('PENDAPATAN ORANG TUA') or ["Rendah","Sedang","Tinggi"])
            tanggungan=st.selectbox("👨‍👩‍👧 Jumlah Tanggungan", pil('JUMLAH TANGGUNGAN') or ["Sedikit","Sedang","Banyak"])
        with c2:
            pekerjaan=st.selectbox("💼 Pekerjaan Orang Tua", pil('PEKERJAAN ORANG TUA') or ["Petani","PNS","Wiraswasta","Buruh"])
            rumah=st.selectbox("🏠 Status Rumah", pil('STATUS RUMAH') or ["Milik Sendiri","Kontrak/Sewa"])
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔍 Prediksi Sekarang", use_container_width=True):
            row={"PENDAPATAN ORANG TUA":pendapatan,"PEKERJAAN ORANG TUA":pekerjaan,
                 "JUMLAH TANGGUNGAN":tanggungan,"STATUS RUMAH":rumah}
            proba=model.predict_proba_log(row); hasil=max(proba,key=proba.get)

            st.markdown("---")
            if hasil=="Ya":
                st.markdown("<div style='text-align:center;padding:1.2rem;'><span class='badge-ya'>✅ LAYAK MENERIMA BSM</span></div>", unsafe_allow_html=True)
                st.success("Siswa ini **LAYAK** menerima Bantuan Siswa Miskin (BSM).")
            else:
                st.markdown("<div style='text-align:center;padding:1.2rem;'><span class='badge-tidak'>❌ TIDAK LAYAK MENERIMA BSM</span></div>", unsafe_allow_html=True)
                st.error("Siswa ini **TIDAK LAYAK** menerima Bantuan Siswa Miskin (BSM).")

            ca,cb=st.columns(2)
            with ca:
                st.markdown(f"<div class='card'><div class='card-title'>📋 Data Input</div>", unsafe_allow_html=True)
                for k,v in row.items():
                    st.markdown(f"<div style='font-size:.9rem;color:{C6};padding:3px 0;'><b>{k}</b> : {v}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with cb:
                st.markdown(f"<div class='card'><div class='card-title'>🧮 Log Posterior</div>", unsafe_allow_html=True)
                for k,v in proba.items():
                    w="#22c55e" if k==hasil else "#f43f5e"
                    label_terpilih=" ← Terpilih" if k==hasil else ""
                    st.markdown(f"<div style='color:{w};font-weight:700;font-size:.9rem;padding:3px 0;'>{k} : {round(v,6)}{label_terpilih}</div>", unsafe_allow_html=True)
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

        def pil2(kolom):
            return sorted(df_ref[kolom].unique().tolist()) if kolom in df_ref.columns else []

        tab1, tab2, tab3 = st.tabs(["➕ Tambah Manual", "📂 Import File", "📊 Hasil Uji"])

        # TAB 1 — TAMBAH MANUAL
        with tab1:
            st.markdown(f"<div class='card'><div class='card-title'>✏️ Input Data Baru</div>", unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1:
                t_pend=st.selectbox("💰 Pendapatan", pil2('PENDAPATAN ORANG TUA') or ["Rendah","Sedang","Tinggi"], key="du1")
                t_tang=st.selectbox("👨‍👩‍👧 Tanggungan", pil2('JUMLAH TANGGUNGAN') or ["Sedikit","Sedang","Banyak"], key="du2")
            with c2:
                t_pek=st.selectbox("💼 Pekerjaan", pil2('PEKERJAAN ORANG TUA') or ["Petani","PNS","Wiraswasta"], key="du3")
                t_rmh=st.selectbox("🏠 Status Rumah", pil2('STATUS RUMAH') or ["Milik Sendiri","Kontrak/Sewa"], key="du4")
            st.markdown("</div>", unsafe_allow_html=True)

            ca,cb,cc=st.columns(3)
            with ca:
                if st.button("🔍 Prediksi & Tambah", use_container_width=True):
                    row={"PENDAPATAN ORANG TUA":t_pend,"PEKERJAAN ORANG TUA":t_pek,
                         "JUMLAH TANGGUNGAN":t_tang,"STATUS RUMAH":t_rmh}
                    hasil=model.predict_one(row); row[TARGET]=hasil
                    new_row=pd.DataFrame([row])
                    st.session_state.data_uji=pd.concat([st.session_state.data_uji, new_row], ignore_index=True)
                    if hasil=="Ya": st.success(f"✅ Ditambahkan — Prediksi: **{hasil}**")
                    else: st.error(f"❌ Ditambahkan — Prediksi: **{hasil}**")

            with cb:
                if st.button("🗑 Hapus Data Terakhir", use_container_width=True):
                    if len(st.session_state.data_uji)>0:
                        st.session_state.data_uji=st.session_state.data_uji.iloc[:-1].reset_index(drop=True)
                        st.warning("Data terakhir dihapus.")

            with cc:
                if st.button("🗑 Hapus Semua", use_container_width=True):
                    st.session_state.data_uji=pd.DataFrame()
                    st.warning("Semua data uji dihapus.")

        # TAB 2 — IMPORT FILE
        with tab2:
            st.info("Upload file data uji (tanpa kolom LABEL). Sistem akan memprediksi otomatis.")
            file_uji=st.file_uploader("Upload File Data Uji", type=["csv","xlsx"], key="fu")

            if file_uji and st.button("📂 Import & Prediksi Semua"):
                try:
                    df_imp=baca_file(file_uji)
                    df_clean=auto_preprocess(df_imp)

                    # Jika tidak ada kolom LABEL, prediksi semua
                    if 'LABEL' not in df_clean.columns:
                        fitur_ada=[f for f in FITUR if f in df_clean.columns]
                        df_clean['LABEL']=model.predict(df_clean[fitur_ada])

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
                st.dataframe(df_u, use_container_width=True, height=420)

                # Simpan ke session hasil_uji
                st.session_state.hasil_uji=st.session_state.data_uji.copy()

                st.markdown("---")
                ca,cb=st.columns(2)
                with ca:
                    st.download_button("📥 Download Data Uji CSV",
                        data=df_to_csv_bytes(df_u), file_name="data_uji_bsm.csv",
                        mime="text/csv", use_container_width=True)
                with cb:
                    st.download_button("📥 Download Data Uji Excel",
                        data=df_to_excel_bytes(df_u), file_name="data_uji_bsm.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True)

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

        # ---- 1. Dataset Preprocessing ----
        st.markdown(f"<div class='card'><div class='card-title'>🔬 1. Dataset Hasil Preprocessing</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:.88rem;color:{C6};margin-bottom:.8rem;'>Dataset setelah normalisasi — {len(df)} data, {len(df.columns)} kolom.</div>", unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.download_button("📥 CSV — Preprocessing",
                data=df_to_csv_bytes(df), file_name="01_preprocessing_bsm.csv",
                mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("📥 Excel — Preprocessing",
                data=df_to_excel_bytes(df), file_name="01_preprocessing_bsm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- 2. Hasil Prediksi Training ----
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
                    mime="text/csv", use_container_width=True)
            with c2:
                st.download_button("📥 Excel — Hasil Prediksi Training",
                    data=df_to_excel_bytes(df_pred), file_name="02_hasil_prediksi_training.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ---- 3. Laporan Evaluasi ----
            st.markdown(f"<div class='card'><div class='card-title'>📋 3. Laporan Evaluasi Model</div>", unsafe_allow_html=True)
            labels=sorted(set(y_true))
            cm=confusion_matrix(y_true,y_pred,labels=labels)
            rpt=pd.DataFrame(classification_report(y_true,y_pred,output_dict=True)).transpose().round(4)

            # Tambah confusion matrix ke laporan
            cm_df=pd.DataFrame(cm,
                index=[f"Aktual_{l}" for l in labels],
                columns=[f"Prediksi_{l}" for l in labels])

            # Gabung jadi satu sheet ringkasan
            ringkasan=pd.DataFrame({
                "Metrik":["Akurasi","Total Data","Prediksi Benar","Prediksi Salah"],
                "Nilai":[f"{round(acc*100,2)}%", len(y_true),
                         sum(a==p for a,p in zip(y_true,y_pred)),
                         sum(a!=p for a,p in zip(y_true,y_pred))]
            })

            # Excel multi-sheet
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
                    mime="text/csv", use_container_width=True)
            with c2:
                st.download_button("📥 Excel — Laporan Lengkap (Multi-Sheet)",
                    data=buf.getvalue(), file_name="03_laporan_evaluasi_lengkap.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---- 4. Data Uji ----
        if not st.session_state.data_uji.empty:
            df_u=st.session_state.data_uji.copy()
            df_u.insert(0,"No",range(1,len(df_u)+1))

            st.markdown(f"<div class='card'><div class='card-title'>📝 4. Hasil Data Uji</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:.88rem;color:{C6};margin-bottom:.8rem;'>{len(df_u)} data uji tersedia.</div>", unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1:
                st.download_button("📥 CSV — Data Uji",
                    data=df_to_csv_bytes(df_u), file_name="04_data_uji_bsm.csv",
                    mime="text/csv", use_container_width=True)
            with c2:
                st.download_button("📥 Excel — Data Uji",
                    data=df_to_excel_bytes(df_u), file_name="04_data_uji_bsm.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("💡 Belum ada data uji. Tambahkan di menu **📝 Data Uji** untuk mengaktifkan download ini.")

# =========================================================
# ℹ️ TENTANG SISTEM
# =========================================================

elif menu == "ℹ️ Tentang Sistem":

    st.markdown("## ℹ️ Tentang Sistem")

    # --- Hero Banner ---
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
        # Deskripsi Sistem
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

        # Fitur Unggulan
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
        # Identitas Pengembang
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

        # Teknologi yang Digunakan
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

        # Atribut Klasifikasi
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>📌 Atribut Klasifikasi</div>
            <div style='font-size:.85rem;color:{C6};line-height:2.1;'>
            <span style='color:{C4};font-weight:800;'>1.</span> Pendapatan Orang Tua<br>
            <span style='color:{C4};font-weight:800;'>2.</span> Pekerjaan Orang Tua<br>
            <span style='color:{C4};font-weight:800;'>3.</span> Jumlah Tanggungan<br>
            <span style='color:{C4};font-weight:800;'>4.</span> Status Rumah<br>
            <span style='color:#22c55e;font-weight:800;'>→</span>
            <b>Label :</b> Ya / Tidak
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Panduan Penggunaan
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

    # Footer
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

