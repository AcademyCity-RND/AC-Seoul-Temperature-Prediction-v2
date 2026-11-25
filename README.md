# Seoul Next-Day Maximum Temperature Predictor (Streamlit)

Versi **Streamlit** dari tugas PPD Pertemuan XI — *Model Deployment*. Prediksi suhu maksimum besok di Seoul dari 21 variabel cuaca/topografi hari ini.

Proyek Flask referensi: `../AC-Seoul-Temperature-Prediction`. Versi ini dibuat sebagai folder terpisah — **kode Flask asli tidak diubah**.

---

## Fitur

- **Hero fullscreen** (`100dvh`) dengan ringkasan model, lalu **form section fullscreen** berikutnya (scroll-snap per section).
- **21 input** dalam dua kolom (11 kiri, 10 kanan) — muat dalam satu layar hingga 1366×768 tanpa scroll.
- **Popup hasil** memakai `st.dialog` dengan animasi fade in/out + backdrop blur.
- **Sidebar**: penjelasan, import CSV, reset ke nilai contoh, sumber data.
- Tema **dark modern-minimalis** lewat `.streamlit/config.toml` + CSS ringan di `app.py` (tanpa dependensi UI tambahan).
- Format angka Indonesia (koma desimal), label awam + nama variabel dalam kurung.

---

## Struktur Proyek

```
AC-Seoul-Temperature-Prediction-v2/
├── app.py                    # Aplikasi Streamlit (536 baris)
├── rf_model.joblib           # Random Forest Regressor · 48 MB · 21 fitur
├── feature_names.json        # Urutan 21 nama fitur
├── sample_input.json         # Nilai default formulir (1 baris contoh dataset)
├── contoh_input_cuaca.csv    # Contoh format CSV untuk Import CSV
├── Bias_correction_ucl.csv   # Dataset training (UCI ID 514) · 1,6 MB
├── requirements.txt          # Dependensi (versi terpin)
├── .streamlit/config.toml    # Tema gelap + headless
├── 1-halaman-input.png       # Screenshot hero + form
├── 2-halaman-output.png      # Screenshot hasil prediksi
└── README.md
```

---

## Menjalankan Secara Lokal

Butuh **Python 3.12** (disarankan).

```bash
cd AC-Seoul-Temperature-Prediction-v2

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`. Untuk port lain:

```bash
streamlit run app.py --server.port 8502
```

Semua path file (`rf_model.joblib`, `feature_names.json`, `sample_input.json`)
**relatif terhadap `app.py`** — tidak ada path absolut Windows, tidak ada secret/API key.
Jadi aman dipindah ke mesin/OS lain.

---

## Deploy ke Streamlit Community Cloud (gratis)

1. Push **folder ini sebagai repo GitHub tersendiri** (disarankan) atau sebagai subfolder
   dari repo besar. Kalau subfolder, isi *Main file path* di langkah 3 dengan
   `AC-Seoul-Temperature-Prediction-v2/app.py`.
2. Buka <https://share.streamlit.io> → **New app** → pilih repo & branch.
3. *Main file path*: `app.py` (atau path subfolder seperti di atas).
4. *Python version*: **3.12** (Advanced settings).
5. **Deploy.** URL publik: `https://<user>-<repo>-<hash>.streamlit.app`.

### Yang perlu diperhatikan saat deploy

- `requirements.txt` sudah **terpin versinya** — jangan dihapus, Streamlit Cloud
  menginstall dari situ. Versi teruji: Streamlit 1.67.0, pandas 3.1.1,
  scikit-learn 1.9.1, joblib 1.6.0, numpy 2.2.6.
- `rf_model.joblib` berukuran **~48 MB**. Masih aman di GitHub (limit 100 MB/file)
  dan di resource Streamlit Cloud (1 GB RAM), tapi repo bisa agak lama di-clone.
  Tidak wajib Git LFS.
- `.streamlit/config.toml` sudah berisi `headless = true` dan theme gelap —
  **tidak perlu setting tambahan** di dashboard Streamlit Cloud.
- Tidak ada environment variable / secret yang perlu diisi.
- Free tier: app public repo unlimited, 1 private app, 1 GB RAM / 1 vCPU,
  **sleep setelah ~7 hari idle** (cold start 30–60 detik). Custom domain butuh
  plan berbayar — alternatif gratis: **Cloudflare Tunnel** ke `localhost:8501`.

### Deploy alternatif (custom domain gratis)

Jalankan Streamlit di VPS/PC lalu expose via Cloudflare Tunnel:

```bash
cloudflared tunnel create seoul-predictor
cloudflared tunnel route dns seoul-predictor seoul.namadomain.com
cloudflared tunnel run seoul-predictor --url http://localhost:8501
```

Tambahkan di `.streamlit/config.toml` kalau lewat reverse proxy:

```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
```

---

## Dataset & Model

- **Dataset**: UCI ML Repository ID 514 — *Bias Correction of Numerical Prediction
  Model Temperature Forecast* (DOI `10.1029/2019EA000740`). Observasi stasiun
  Seoul, Korea Selatan (lat 37.4562–37.6450, lon 126.8260–127.1350).
- **Target**: `Next_Tmax` — suhu maksimum hari berikutnya (°C).
- **Fitur (21)**: `Present_Tmax`, `Present_Tmin`, 14 variabel LDAPS (kelembapan,
  lapse rate, angin, panas laten, tutupan awan, curah hujan), `lat`, `lon`, `DEM`,
  `Slope`, `Solar radiation`.
- **Model**: Random Forest Regressor, **R² = 0,8942** (data uji).

## Alur Kode (modul Pertemuan XI)

1. `st.set_page_config(...)` — judul tab, ikon, layout lebar.
2. CSS + hero section `100dvh` — identitas aplikasi & ringkasan model.
3. Sidebar — penjelasan, **Import CSV**, **Reset ke nilai contoh**, sumber data.
4. Form section `100dvh` — 21 input dalam 2 kolom (11 kiri, 10 kanan).
5. `st.form_submit_button("Prediksi Suhu Besok")`.
6. Hasil → `st.dialog` popup: `st.metric` prediksi + delta terhadap `Present_Tmax`,
   metadata model, dan ringkasan input.

Nilai default diambil dari `sample_input.json`. Tombol reset menulis ulang
`st.session_state` lalu `st.rerun()`.

## Cara Pakai

- **Manual**: isi 21 field di form → klik **Prediksi Suhu Besok**.
- **CSV**: siapkan CSV satu baris dengan kolom sesuai `feature_names.json`
  (lihat `contoh_input_cuaca.csv`) → Upload di sidebar → form terisi otomatis.

## Hasil Uji

Input contoh (`sample_input.json`) menghasilkan:

```
Prediksi Suhu Maksimum Besok = 29,65 °C   (Δ +0,95 °C dari hari ini)
```

Nilai ini identik dengan hasil versi Flask — konsisten antar kedua deployment.

