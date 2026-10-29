# 🏭 Maintenance Copilot

Sistem deteksi dini kerusakan mesin berbasis **machine learning hybrid** (One-Class SVM + XGBoost), dilengkapi penjelasan **SHAP** dan saran tindakan otomatis dari **Gemini AI**. Dibangun di atas dataset AI4I 2020 Predictive Maintenance

Link deployment website Streamlit (belum backend) : https://sistem-pemeliharaan-mesin-gnuaxjqqhxdtw9qgl8hwne.streamlit.app/

## ✨ Fitur Utama

- **Dashboard Monitoring** – ringkasan status seluruh mesin (Normal / Warning / Critical), filter berdasarkan status dan penyebab, serta detail sensor per mesin.
- **Cek Mesin per Product ID** – masukkan ID mesin, sistem mengambil data sensor dan menampilkan diagnosa secara langsung.
- **Deteksi Hybrid** – skor anomali dari One-Class SVM dijadikan fitur tambahan untuk klasifikasi XGBoost.
- **Penjelasan SHAP (Waterfall)** – grafik kontribusi tiap sensor terhadap prediksi, sehingga hasil model bisa dijelaskan.
- **Saran Tindakan AI** – analisis akar masalah dan instruksi teknisi dari Gemini, dengan fallback saran manual bila AI tidak tersedia.
- **Chatbot** – endpoint tanya jawab seputar perawatan mesin.

## 🧠 Cara Kerja

```
Data sensor ─► Aturan fisika (PWF, HDF, OSF, TWF)
     │
     ├─► One-Class SVM ─► Anomaly Score ─┐
     │                                    ▼
     └──────────────────────────► XGBoost (hybrid) ─► Probabilitas gagal
                                          │
                                          ├─► SHAP waterfall
                                          └─► Gemini ─► Saran tindakan
```

**Penentuan status**

| Status | Kondisi |
|---|---|
| Critical | Mesin Rusak |
| Warning | Mesin Naomali / Akan Rusak |
| Normal | Mesin Normal |

**Aturan yang dicek**

- **PWF** (Power Failure): daya < 3500 W atau > 9000 W
- **HDF** (Heat Dissipation Failure): selisih suhu < 8,6 K dan kecepatan < 1380 rpm
- **OSF** (Overstrain Failure): torque × tool wear > 11000
- **TWF** (Tool Wear Failure): tool wear ≥ 200 menit

## 🚀 Menjalankan Secara Lokal

**1. Instal dependensi**

```bash
pip install flask flask-cors streamlit requests altair pandas numpy \
            scikit-learn xgboost shap matplotlib joblib google-genai
```

> Gunakan versi `scikit-learn` dan `xgboost` yang sama dengan saat model dilatih, karena file `.pkl` sensitif terhadap versi.

**2. Atur kunci API Gemini** (opsional; tanpa ini saran AI memakai fallback manual)

```bash
# Linux / macOS
export GEMINI_API_KEY="kunci_anda"

# Windows (cmd)
set GEMINI_API_KEY=kunci_anda
```

**3. Jalankan backend** (port 5000)

```bash
python backend_api.py
```

**4. Jalankan frontend** (terminal lain)

```bash
streamlit run frontend.py
```

Buka alamat yang ditampilkan Streamlit (biasanya `http://localhost:8501`), lalu pilih **Dashboard Monitoring** atau **Cek Mesin Manual**.


Penulis

Jesika Yudiani Putri | Project Capstone Asah by Dicoding
