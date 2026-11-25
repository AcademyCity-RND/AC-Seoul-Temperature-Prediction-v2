"""
Aplikasi Streamlit — Seoul Next-Day Maximum Temperature Predictor
PPD Pertemuan XI — Model Deployment (Streamlit), versi Tugas

Dataset : UCI ID 514 "Bias Correction of Numerical Prediction Model
          Temperature Forecast" (Bias_correction_ucl.csv) — observasi Seoul, Korea Selatan.
Model   : Random Forest Regressor (rf_model.joblib), R² = 0.8942, 21 fitur.

Layout  : hero section 100dvh -> form section 100dvh (scroll-snap per section),
          hasil prediksi tampil sebagai popup (st.dialog) dengan animasi fade.
"""

import json
import os
from typing import Optional

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# 1. Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title="Seoul Temperature Predictor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. Tema gelap, section fullscreen, animasi popup
# ============================================================
st.markdown(
    """
    <style>
    /* --- chrome Streamlit: JANGAN sembunyikan stToolbar (tombol sidebar ada di sana) --- */
    #MainMenu, footer, [data-testid="stDecoration"],
    [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] {display: none !important;}
    header[data-testid="stHeader"] {background: transparent !important;}

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
    }
    html {scroll-behavior: smooth;}

    /* --- scroll container: snap per section --- */
    section[data-testid="stMain"] {
        scroll-snap-type: y proximity;
        scrollbar-width: thin;
        scrollbar-color: #24334d transparent;
    }
    .block-container {
        max-width: 1280px;
        padding: 0 2rem 1rem !important;
    }
    .block-container > div[data-testid="stVerticalBlock"] {gap: 0 !important;}

    /* --- HERO: satu layar penuh --- */
    .hero-wrap {
        min-height: 100dvh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        position: relative;
        padding: 2rem 1rem;
        scroll-snap-align: start;
        scroll-snap-stop: always;
    }
    .hero-glow {
        position: absolute; inset: 0; pointer-events: none;
        background:
            radial-gradient(760px 340px at 50% 8%, rgba(56,189,248,.16), transparent 70%),
            radial-gradient(560px 300px at 50% 100%, rgba(37,99,235,.12), transparent 70%);
    }
    .hero-inner {position: relative; z-index: 1; max-width: 900px;}
    .hero-eyebrow {
        font-size: .7rem; letter-spacing: .2em; text-transform: uppercase;
        color: #7dd3fc; font-weight: 700; margin-bottom: 1rem;
    }
    .hero-title {
        font-size: clamp(2.4rem, 5.4vw, 4.15rem);
        line-height: 1.04; letter-spacing: -.035em;
        margin: 0 0 .9rem; color: #f8fafc; font-weight: 800;
    }
    .hero-title em {font-style: normal; color: #38bdf8;}
    .hero-sub {
        font-size: clamp(.95rem, 1.2vw, 1.1rem);
        color: #a3b1c6; line-height: 1.55; max-width: 60ch; margin: 0 auto 1.9rem;
    }
    .hero-chips {display: flex; flex-wrap: wrap; gap: .5rem; justify-content: center; margin-bottom: 2.4rem;}
    .hero-chip {
        font-size: .73rem; color: #cbd5e1; border: 1px solid #24334d;
        background: rgba(13,21,38,.75); border-radius: 999px; padding: .32rem .8rem;
        backdrop-filter: blur(6px);
    }
    .hero-chip b {color: #7dd3fc; font-weight: 600;}
    .hero-cta {
        display: inline-flex; align-items: center; gap: .5rem;
        font-size: .82rem; font-weight: 650; letter-spacing: .02em;
        color: #e8eef7; border: 1px solid #2b3d5c; background: #101c30;
        border-radius: 999px; padding: .6rem 1.25rem; cursor: pointer;
        transition: all .18s ease; user-select: none;
    }
    .hero-cta:hover {border-color: #38bdf8; color: #7dd3fc; transform: translateY(-1px);}
    .hero-scroll {
        position: absolute; bottom: 1.6rem; left: 50%; transform: translateX(-50%);
        font-size: .64rem; letter-spacing: .18em; text-transform: uppercase;
        color: #4f6787; z-index: 1;
    }
    .hero-scroll span {display:block; margin-top:.35rem; font-size: 1rem; color:#38bdf8; animation: bob 1.8s ease-in-out infinite;}
    @keyframes bob {0%,100%{transform:translateY(0);opacity:.55} 50%{transform:translateY(5px);opacity:1}}

    /* --- FORM: satu layar penuh, tanpa scroll internal --- */
    div[data-testid="stForm"] {
        min-height: calc(100dvh - 1rem);
        display: flex; flex-direction: column; justify-content: center;
        border: 1px solid #1b2740 !important;
        border-radius: 18px !important;
        padding: 1.5rem 1.75rem 1.25rem !important;
        background: linear-gradient(180deg, #0e1727 0%, #0a1120 100%) !important;
        scroll-snap-align: start;
        scroll-snap-stop: always;
    }
    div[data-testid="stForm"] > div[data-testid="stVerticalBlock"] {gap: .5rem !important;}

    .form-head {margin-bottom: .9rem;}
    .form-head h2 {
        font-size: 1.3rem; margin: 0 0 .2rem; color: #f1f5f9;
        font-weight: 700; letter-spacing: -.02em;
    }
    .form-head p {margin: 0; font-size: .8rem; color: #7f93ad;}

    /* input kompak supaya 21 field muat satu layar */
    div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {gap: .18rem !important;}
    div[data-testid="stElementContainer"]:has(div[data-testid="stNumberInput"]) {height: auto !important;}
    div[data-testid="stNumberInput"] {margin-bottom: 0 !important;}
    div[data-testid="stNumberInput"] > div {
        background: #0a1424 !important;
        border: 1px solid #1e2c45 !important;
        border-radius: 8px !important;
        min-height: 1.55rem !important;
    }
    div[data-testid="stNumberInput"] > div:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px rgba(56,189,248,.3);
    }
    div[data-testid="stNumberInput"] input {
        height: 1.62rem !important; font-size: .8rem !important;
        color: #e8eef7 !important; background: transparent !important;
        padding: 0 .45rem !important; line-height: 1 !important;
    }
    div[data-testid="stNumberInput"] button {
        height: 1.62rem !important; width: 1.5rem !important; min-height: 0 !important;
        background: #101d33 !important; color: #7dd3fc !important;
        border: none !important; font-weight: 700 !important; font-size: .78rem !important;
    }
    div[data-testid="stNumberInput"] button:hover {background: #16263f !important;}

    div[data-testid="stWidgetLabel"] {margin-bottom: 0 !important; min-height: 0 !important; height: auto !important;}
    div[data-testid="stWidgetLabel"] p {
        font-size: .62rem !important; line-height: 1 !important;
        color: #93a6c0 !important; font-weight: 500 !important;
        margin: 0 !important;
    }
    div[data-testid="stNumberInputContainer"] {min-height: 0 !important; height: 22px !important; padding-top: 0 !important; padding-bottom: 0 !important; display: flex !important; align-items: center !important;}
    div[data-testid="stNumberInputContainer"] > div:first-of-type {min-height: 0 !important; height: 22px !important;}
    div[data-testid="stNumberInput"] input {min-height: 22px !important; height: 22px !important;}

    /* tombol submit */
    div[data-testid="stFormSubmitButton"] > button {
        border: none !important; border-radius: 10px !important;
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
        color: #fff !important; font-weight: 700 !important;
        padding: .6rem 1.3rem !important; margin-top: .55rem;
        box-shadow: 0 10px 26px rgba(37,99,235,.28);
        transition: all .18s ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        filter: brightness(1.09); transform: translateY(-1px);
        box-shadow: 0 14px 32px rgba(37,99,235,.38);
    }

    /* layar pendek (laptop 768px): rapatkan lagi supaya tetap satu layar */
    @media (max-height: 830px) {
        div[data-testid="stForm"] {padding: .9rem 1.35rem .7rem !important;}
        .form-head {margin-bottom: .4rem;}
        .form-head h2 {font-size: 1.08rem;}
        .form-head p {font-size: .7rem;}
        div[data-testid="stFormSubmitButton"] > button {margin-top: .25rem; padding: .42rem 1rem !important;}
    }
    @media (max-height: 700px) {
        div[data-testid="stForm"] {padding: .6rem 1.1rem .5rem !important;}
        div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {gap: .1rem !important;}
        .form-head {margin-bottom: .25rem;}
        .form-head h2 {font-size: .98rem;}
        .form-head p {display: none;}
    }

    /* --- sidebar --- */
    section[data-testid="stSidebar"] {
        background: #080f1d; border-right: 1px solid #16213a;
    }
    section[data-testid="stSidebar"] hr {border-color: #16213a;}
    /* tombol buka sidebar saat tertutup tetap terlihat */
    div[data-testid="stExpandSidebarButton"] {
        color: #cbd5e1 !important;
        background: #0d1526 !important;
        border: 1px solid #1e2c45 !important;
        border-radius: 9px !important;
        box-shadow: 0 6px 18px rgba(0,0,0,.35);
    }
    div[data-testid="stExpandSidebarButton"]:hover {background: #14223a !important;}
    div[data-testid="stSidebarCollapseButton"] button {color: #7f93ad !important;}
    div[data-testid="stSidebarCollapseButton"] button:hover {color: #7dd3fc !important;}

    /* --- POPUP hasil prediksi: fade in / fade out --- */
    @keyframes dlgBackdropIn  {from {opacity: 0; backdrop-filter: blur(0);} to {opacity: 1; backdrop-filter: blur(6px);}}
    @keyframes dlgBackdropOut {from {opacity: 1; backdrop-filter: blur(6px);} to {opacity: 0; backdrop-filter: blur(0);}}
    @keyframes dlgCardIn {
        from {opacity: 0; transform: translateY(22px) scale(.965);}
        to   {opacity: 1; transform: translateY(0) scale(1);}
    }
    @keyframes dlgCardOut {
        from {opacity: 1; transform: translateY(0) scale(1);}
        to   {opacity: 0; transform: translateY(12px) scale(.985);}
    }
    /* lapisan gelap full-screen */
    div[data-testid="stDialog"] {
        background: rgba(3,7,15,.68) !important;
        backdrop-filter: blur(6px);
        animation: dlgBackdropIn .32s ease both;
        align-items: center !important;
        justify-content: center !important;
    }
    /* kartu popup */
    div[data-testid="stDialog"] section[role="dialog"] {
        animation: dlgCardIn .4s cubic-bezier(.22,1,.36,1) both;
        border: 1px solid #1e3a5f !important;
        border-radius: 20px !important;
        background:
            radial-gradient(620px 220px at 85% -30%, rgba(56,189,248,.18), transparent 65%),
            linear-gradient(160deg, #0e1c33 0%, #0a1120 75%) !important;
        box-shadow: 0 30px 80px rgba(0,0,0,.6);
        padding: 1.5rem 1.6rem 1.2rem !important;
    }
    /* animasi keluar bila runtime menahan elemen saat exit */
    div[data-testid="stDialog"][data-exiting],
    div[data-testid="stDialog"]:has(section[data-exiting]) {
        animation: dlgBackdropOut .22s ease both !important;
    }
    div[data-testid="stDialog"] section[role="dialog"][data-exiting] {
        animation: dlgCardOut .22s ease both !important;
    }
    /* sembunyikan tombol X bawaan; penutup = tombol Tutup */
    div[data-testid="stDialog"] section[role="dialog"] > button[aria-label="Close"] {
        display: none !important;
    }
    div[data-testid="stDialog"] section[role="dialog"] h2,
    div[data-testid="stDialog"] section[role="dialog"] [data-testid="stHeading"] {
        font-size: 1.02rem !important;
        margin-bottom: .55rem !important;
    }

    .pop-label {
        font-size: .7rem; letter-spacing: .16em; text-transform: uppercase;
        color: #7dd3fc; font-weight: 700; margin-bottom: .5rem;
    }
    .pop-value {
        font-size: clamp(2.6rem, 7vw, 4rem); font-weight: 800; line-height: 1;
        letter-spacing: -.035em; color: #f8fafc;
    }
    .pop-value small {font-size: 1.35rem; color: #93a6c0; font-weight: 500; margin-left: .2rem;}
    .pop-delta {font-size: .86rem; font-weight: 650; margin-top: .7rem;}
    .pop-delta.up {color: #fbbf24;}
    .pop-delta.down {color: #34d399;}
    .pop-meta {
        display: flex; flex-wrap: wrap; gap: 1.2rem;
        margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid #1e3a5f;
        font-size: .85rem; color: #b6c6d8;
    }
    .pop-meta b {color: #f8fafc; font-weight: 700;}
    .pop-meta span + span {border-left: 1px solid #2a4566; padding-left: 1.2rem;}
    /* tombol Tutup di dalam popup — primary accent */
    div[data-testid="stDialog"] section[role="dialog"] div[data-testid="stButton"] button {
        border: none !important;
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: .8rem !important;
        padding: .55rem 1.3rem !important;
        margin-top: .75rem !important;
        box-shadow: 0 6px 22px rgba(37,99,235,.35);
        transition: all .16s ease;
    }
    div[data-testid="stDialog"] section[role="dialog"] div[data-testid="stButton"] button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 10px 28px rgba(37,99,235,.45);
    }
    div[data-testid="stDialog"] section[role="dialog"] div[data-testid="stButton"] button:focus-visible {
        outline: 2px solid #38bdf8; outline-offset: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. Data, model, label
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.json")
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_input.json")

with open(FEATURES_PATH, encoding="utf-8") as f:
    FEATURE_NAMES = json.load(f)

with open(SAMPLE_PATH, encoding="utf-8") as f:
    SAMPLE_INPUT = json.load(f)

FEATURE_LABELS = {
    "Present_Tmax": "Suhu Maksimum Hari Ini (Present_Tmax, °C)",
    "Present_Tmin": "Suhu Minimum Hari Ini (Present_Tmin, °C)",
    "LDAPS_RHmin": "Kelembapan Minimum (LDAPS_RHmin, %)",
    "LDAPS_RHmax": "Kelembapan Maksimum (LDAPS_RHmax, %)",
    "LDAPS_Tmax_lapse": "Laju Penurunan Suhu Maksimum (LDAPS_Tmax_lapse, °C)",
    "LDAPS_Tmin_lapse": "Laju Penurunan Suhu Minimum (LDAPS_Tmin_lapse, °C)",
    "LDAPS_WS": "Kecepatan Angin (LDAPS_WS, m/s)",
    "LDAPS_LH": "Panas Laten (LDAPS_LH, W/m²)",
    "LDAPS_CC1": "Tutupan Awan Lapis 1 (LDAPS_CC1, fraksi 0-1)",
    "LDAPS_CC2": "Tutupan Awan Lapis 2 (LDAPS_CC2, fraksi 0-1)",
    "LDAPS_CC3": "Tutupan Awan Lapis 3 (LDAPS_CC3, fraksi 0-1)",
    "LDAPS_CC4": "Tutupan Awan Lapis 4 (LDAPS_CC4, fraksi 0-1)",
    "LDAPS_PPT1": "Curah Hujan Periode 1 (LDAPS_PPT1, mm)",
    "LDAPS_PPT2": "Curah Hujan Periode 2 (LDAPS_PPT2, mm)",
    "LDAPS_PPT3": "Curah Hujan Periode 3 (LDAPS_PPT3, mm)",
    "LDAPS_PPT4": "Curah Hujan Periode 4 (LDAPS_PPT4, mm)",
    "lat": "Lintang Lokasi (lat, derajat)",
    "lon": "Bujur Lokasi (lon, derajat)",
    "DEM": "Ketinggian Permukaan (DEM, m)",
    "Slope": "Kemiringan Lereng (Slope, derajat)",
    "Solar radiation": "Radiasi Matahari (Solar radiation, Wh/m²)",
}

FEATURE_BOUNDS = {
    "Present_Tmax": (10.0, 45.0, 0.1),
    "Present_Tmin": (0.0, 35.0, 0.1),
    "LDAPS_RHmin": (0.0, 100.0, 0.1),
    "LDAPS_RHmax": (0.0, 100.0, 0.1),
    "LDAPS_Tmax_lapse": (10.0, 45.0, 0.1),
    "LDAPS_Tmin_lapse": (5.0, 35.0, 0.1),
    "LDAPS_WS": (0.0, 30.0, 0.1),
    "LDAPS_LH": (-50.0, 250.0, 0.1),
    "LDAPS_CC1": (0.0, 1.0, 0.01),
    "LDAPS_CC2": (0.0, 1.0, 0.01),
    "LDAPS_CC3": (0.0, 1.0, 0.01),
    "LDAPS_CC4": (0.0, 1.0, 0.01),
    "LDAPS_PPT1": (0.0, 30.0, 0.1),
    "LDAPS_PPT2": (0.0, 30.0, 0.1),
    "LDAPS_PPT3": (0.0, 30.0, 0.1),
    "LDAPS_PPT4": (0.0, 30.0, 0.1),
    "lat": (37.40, 37.70, 0.0001),
    "lon": (126.80, 127.20, 0.0001),
    "DEM": (0.0, 250.0, 0.1),
    "Slope": (0.0, 10.0, 0.01),
    "Solar radiation": (4000.0, 6500.0, 1.0),
}

DEFAULTS = {fn: SAMPLE_INPUT[fn] for fn in FEATURE_NAMES}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource(show_spinner="Memuat model Random Forest...")
def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def _to_float(x: Optional[float]) -> float:
    try:
        return float(x) if x is not None else float("nan")
    except Exception:
        return float("nan")


def _render_number_input(fn: str):
    lo, hi, step = FEATURE_BOUNDS[fn]
    return st.number_input(
        FEATURE_LABELS[fn],
        min_value=lo,
        max_value=hi,
        step=step,
        format="%.4f" if step < 0.01 else "%.2f",
        key=fn,
    )


# ============================================================
# 4. Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### Panel Kontrol")
    st.caption(
        "Isi 21 variabel cuaca, lalu jalankan prediksi. Nilai awal berasal dari "
        "satu baris contoh dataset Seoul."
    )

    uploaded = st.file_uploader(
        "Import CSV (1 baris, 21 kolom)",
        type=["csv"],
        help="Gunakan contoh_input_cuaca.csv sebagai format acuan.",
    )
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            missing = [c for c in FEATURE_NAMES if c not in df_up.columns]
            if missing:
                st.error(f"Kolom tidak lengkap: {', '.join(missing)}")
            else:
                for fn in FEATURE_NAMES:
                    st.session_state[fn] = float(df_up.iloc[0][fn])
                st.success("CSV berhasil dimuat.")
                st.rerun()
        except Exception as exc:
            st.error(f"Gagal membaca CSV: {exc}")

    if st.button("Reset ke nilai contoh", type="secondary", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    st.divider()
    st.caption("**Sumber data**")
    st.caption(
        "UCI Machine Learning Repository ID 514 — *Bias Correction of Numerical "
        "Prediction Model Temperature Forecast*, observasi Seoul, Korea Selatan."
    )
    st.caption("**Model**")
    st.caption("Random Forest Regressor · 21 fitur · R² = 0,8942 pada data uji.")

# ============================================================
# 5. Hero section — satu layar penuh
# ============================================================
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-glow"></div>
        <div class="hero-inner">
            <div class="hero-eyebrow">Praktikum Penambangan Data · Pertemuan XI</div>
            <h1 class="hero-title">Seoul <em>Next-Day</em><br>Maximum Temperature</h1>
            <p class="hero-sub">Prediksi suhu maksimum besok di Seoul, Korea Selatan
            berdasarkan kondisi cuaca hari ini dan prakiraan model numerik LDAPS.</p>
            <div class="hero-chips">
                <span class="hero-chip">Model <b>Random Forest</b></span>
                <span class="hero-chip">Fitur <b>21</b></span>
                <span class="hero-chip">R² <b>0,8942</b></span>
                <span class="hero-chip">Dataset <b>UCI ID 514</b></span>
                <span class="hero-chip">Lokasi <b>Seoul, KR</b></span>
            </div>
            <div class="hero-cta"
                 onclick="document.querySelector('div[data-testid=&quot;stForm&quot;]').scrollIntoView({behavior:'smooth',block:'start'})">
                Isi Data Cuaca <span>↓</span>
            </div>
        </div>
        <div class="hero-scroll">scroll<span>↓</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 6. Form section — satu layar penuh, tanpa scroll internal
# ============================================================
with st.form("input-form"):
    st.markdown(
        """
        <div class="form-head">
            <h2>Data Cuaca Hari Ini</h2>
            <p>21 variabel · satuan mengikuti dataset · nilai awal = satu baris contoh Seoul</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="medium")
    values = {}

    with left:
        for fn in FEATURE_NAMES[:11]:
            values[fn] = _render_number_input(fn)

    with right:
        for fn in FEATURE_NAMES[11:]:
            values[fn] = _render_number_input(fn)

    submitted = st.form_submit_button("Prediksi Suhu Besok", use_container_width=True)


# ============================================================
# 7. Popup hasil prediksi
# ============================================================
@st.dialog("Hasil Prediksi", width="small")
def show_prediction(prediction: float, delta: float, delta_cls: str):
    pred_txt = f"{prediction:.2f}".replace(".", ",")
    delta_txt = f"{delta:+.2f}".replace(".", ",")
    st.markdown(
        f"""
        <div class="pop-label">Prediksi Suhu Maksimum Besok</div>
        <div class="pop-value">{pred_txt}<small>°C</small></div>
        <div class="pop-delta {delta_cls}">{delta_txt} °C dibanding suhu maksimum hari ini</div>
        <div class="pop-meta">
            <span>Model <b>Random Forest</b></span>
            <span>R² <b>0,8942</b></span>
            <span>Stasiun <b>Seoul</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Tutup", use_container_width=True, key="popup-close"):
        st.rerun()


if submitted:
    with st.spinner("Menghitung prediksi..."):
        model = load_model(MODEL_PATH)
        df_input = pd.DataFrame([{fn: _to_float(values[fn]) for fn in FEATURE_NAMES}])
        prediction = float(model.predict(df_input[FEATURE_NAMES])[0])

    delta = prediction - float(values["Present_Tmax"])
    show_prediction(prediction, delta, "up" if delta >= 0 else "down")
