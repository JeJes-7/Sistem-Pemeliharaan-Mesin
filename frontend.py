# frontend.py (Clean Version)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt

# ==========================================
# 1. KONFIGURASI HALAMAN & BACKEND
# ==========================================
st.set_page_config(
    page_title="Maintenance Copilot",
    page_icon="🏭",
    layout="wide"
)

# Hostname dan port backend
BACKEND_URL = "http://localhost:5000"

# ==========================================
# 2. LOAD DATASET (Untuk Dashboard)
# ==========================================
@st.cache_data
def load_dataset():
    try:
        # Membaca file hasil export dari Notebook
        df = pd.read_csv('data_hasil_model.csv')
        return df
    except FileNotFoundError:
        return None

df_mesin = load_dataset()

# ==========================================
# 3. INISIALISASI STATE
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if "manual_result" not in st.session_state:
    st.session_state.manual_result = None

# ==========================================
# 4. SIDEBAR NAVIGASI
# ==========================================
with st.sidebar:
    st.header("⚙️ Mode Operasi")
    
    # Tombol Navigasi
    if st.button("📊 Dashboard Monitoring", use_container_width=True):
        st.session_state.page = 'dashboard'
        st.rerun()
        
    if st.button("📝 Cek Mesin Manual", use_container_width=True):
        st.session_state.page = 'manual_input'
        st.rerun()

    st.markdown("---")

# ==========================================
# 5. HALAMAN: INPUT MANUAL (REQ KE BACKEND)
# ==========================================
if st.session_state.page == 'manual_input':
    st.title("🩺 Cek Kesehatan Mesin (Live Simulation)")
    st.info("Masukkan parameter sensor di bawah ini untuk diagnosa real-time via Backend.")
    
# --- FORM INPUT ID ---
    with st.container(border=True):
        c_input, c_btn = st.columns([3, 1])
        
        with c_input:
            # Input Text untuk ID
            input_id = st.text_input("Product ID", placeholder="Contoh: L47181, M14860, L47249", help="Masukkan ID mesin yang terdaftar di database.")
        
        with c_btn:
            st.write("") # Spacer agar tombol sejajar
            st.write("")
            btn_analisa = st.button("🔍 Analisa Mesin", type="primary", use_container_width=True)

    # --- LOGIKA ANALISA ---
    if btn_analisa and input_id:
        try:
            # 1. Kirim Request ke Backend (Mode ID)
            with st.spinner(f"Menghubungkan ke mesin {input_id}..."):
                payload = {"mode": "id", "product_id": input_id}
                res = requests.post(f"{BACKEND_URL}/predict", json=payload)
            
            # 2. Proses Response
            if res.status_code == 200:
                result = res.json()
                info = result['machine_info'] # Data sensor asli dari database
                
                st.divider()
                
                # --- BAGIAN A: DATA FISIK MESIN ---
                st.subheader(f"📄 Data Sensor: {input_id}")
                
                # Tampilkan parameter mesin dalam grid
                with st.container(border=True):
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    c1.metric("Tipe", info['Type'])
                    c2.metric("Air Temp", f"{info['Air temperature [K]']} K")
                    c3.metric("Proc. Temp", f"{info['Process temperature [K]']} K")
                    c4.metric("Speed", f"{info['Rotational speed [rpm]']} rpm")
                    c5.metric("Torque", f"{info['Torque [Nm]']} Nm")
                    c6.metric("Tool Wear", f"{info['Tool wear [min]']} min")

                st.write("")

                # --- BAGIAN B: HASIL DIAGNOSA HYBRID ---
                st.subheader("🧠 Hasil Analisa AI")
                
                col_kiri, col_kanan = st.columns([1, 1.5])
                
                with col_kiri:
                    # 1. Status Utama
                    stat = result['status']
                    if stat == "Critical":
                        st.error(f"### 🚨 CRITICAL\nRisiko kerusakan tinggi! ")
                    elif stat == "Warning":
                        st.warning(f"### ⚠️ WARNING\nTerdeteksi anomali.")
                    else:
                        st.success("### ✅ NORMAL\nKondisi mesin optimal.")
                    
                    st.divider()
                    
                    # 2. Analisa Teks Gemini
                    if stat != "Normal":
                        st.info("🤖 **Saran Tindakan (AI Assistant):**")
                        st.markdown(result.get('ai_analysis', 'Tidak ada saran.'))
                    else:
                        st.caption("Mesin beroperasi dalam batas normal. Tidak ada tindakan diperlukan.")

                with col_kanan:
                    # 3. Grafik SHAP (Gambar dari Backend)
                    st.write("**🔍 Faktor Penyebab Utama (SHAP Waterfall):**")
                    
                    if 'shap_image' in result and result['shap_image']:
                        # Decode Base64 ke Gambar
                        import base64
                        image_data = base64.b64decode(result['shap_image'])
                        st.image(image_data, caption="Analisa Kontribusi Sensor (SHAP)", use_container_width=True)
                    else:
                        st.caption("Grafik tidak tersedia untuk status Normal.")

            elif res.status_code == 404:
                st.error(f"❌ Product ID **'{input_id}'** tidak ditemukan di database.")
                st.info("Silakan cek kembali ID pada Dashboard Monitoring.")
            else:
                st.error(f"Terjadi kesalahan pada Backend: {res.text}")
                
        except Exception as e:
            st.error(f"Gagal koneksi ke server: {e}")
            st.warning("Pastikan `backend.py` sudah dijalankan di terminal.")

# ==========================================
# 6. HALAMAN: DASHBOARD MONITORING
# ==========================================
else: 
    st.title("🏭 Dashboard Monitoring Armada")
    
    if df_mesin is None:
        st.error("⚠️ File 'data_hasil_model.csv' tidak ditemukan. Harap export dari Notebook.")
    else:
        # --- A. DATA PROCESSING ---
        @st.cache_data
        def process_dashboard_data(raw_df):
            rows = []
            
            # Mapping Kolom SHAP
            shap_map = {
                'SHAP_Cause_Air_temperature_': 'Air Temp',
                'SHAP_Cause_Process_temperature_': 'Process Temp',
                'SHAP_Cause_Rotational_speed_': 'Rot. Speed',
                'SHAP_Cause_Torque_': 'Torque',
                'SHAP_Cause_Tool_wear_': 'Tool Wear'
            }
            # Mapping Kolom Sensor
            sensor_map = {
                'Air temperature [K]': 'Air Temp',
                'Process temperature [K]': 'Process Temp',
                'Rotational speed [rpm]': 'Rot. Speed',
                'Torque [Nm]': 'Torque',
                'Tool wear [min]': 'Tool Wear'
            }
            
            for _, row in raw_df.iterrows():
                global_stat = row.get('Diagnosis_Global', 'Normal')
                cause = "-"
                
                # Hitung Confidence Dinamis
                s_score = row.get('S', 0)
                if global_stat != 'Normal':
                    pct = min(99.9, 50.0 + abs(s_score))
                else:
                    pct = 0.0
                
                # Tentukan Penyebab dari SHAP
                if global_stat in ['Critical', 'Warning']:
                    found = []
                    for k, v in shap_map.items():
                        if row.get(k, 0) == 1: found.append(v)
                    
                    if found:
                        cause = found[0] # Ambil yang pertama
                    else:
                        cause = "Unknown"
                        
                # Detail Sensor
                s_detail = {}
                for k, v in sensor_map.items():
                    state = "Normal"
                    if global_stat != 'Normal' and v == cause:
                        state = global_stat
                    s_detail[v] = {'val': row.get(k,0), 'state': state}

                rows.append({
                    'Product ID': row['Product ID'],
                    'Type': row['Type'],
                    'Status': global_stat,
                    'Main Cause': cause,
                    'Confidence': pct,
                    'Sensor Detail': s_detail
                })
            
            return pd.DataFrame(rows)

        df_dash = process_dashboard_data(df_mesin)
        
        # --- B. STATISTIK ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Mesin", len(df_dash))
        c2.metric("Normal", len(df_dash[df_dash['Status']=='Normal']))
        c3.metric("Warning", len(df_dash[df_dash['Status']=='Warning']), delta_color="off")
        c4.metric("Critical", len(df_dash[df_dash['Status']=='Critical']), delta_color="inverse")
        
        st.markdown("---")
        
        # --- C. FILTER ---
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_status = st.multiselect("Status", ["Critical", "Warning", "Normal"], default=["Critical", "Warning"])
        with fc2:
            # Dropdown otomatis berisi penyebab yang ada di data
            causes_avail = df_dash[df_dash['Main Cause'] != '-']['Main Cause'].unique()
            sel_cause = st.multiselect("Penyebab", causes_avail)
        with fc3:
            # Toggle Filter Spesifik
            show_torque = st.checkbox("Hanya Masalah Torque")

        # Apply Filter
        df_view = df_dash.copy()
        if sel_status: df_view = df_view[df_view['Status'].isin(sel_status)]
        if sel_cause: df_view = df_view[df_view['Main Cause'].isin(sel_cause)]
        if show_torque: 
            df_view = df_view[df_view['Main Cause'] == 'Torque']
            
        st.info(f"Menampilkan {len(df_view)} data.")
        
        # --- D. VISUALISASI KARTU ---
        limit = 20
        df_show = df_view.head(limit)
        
        cols = st.columns(3)
        for idx, row in df_show.iterrows():
            with cols[idx % 3]:
                with st.container(border=True):
                    # Judul & Status
                    h1, h2 = st.columns([2,1])
                    h1.write(f"**{row['Product ID']}** ({row['Type']})")
                    
                    stat = row['Status']
                    color = "red" if stat == 'Critical' else "orange" if stat == 'Warning' else "green"
                    h2.markdown(f":{color}[**{stat.upper()}**]")
                    
                    st.divider()
                    
                    # Penyebab & Confidence Dinamis
                    if stat != 'Normal':
                        st.write(f"🛑 **{row['Main Cause']}**")
                        st.progress(int(row['Confidence']), text=f"Confidence: {row['Confidence']:.1f}%")
                    else:
                        st.write("✅ Optimal")
                    
                    # Expander
                    with st.expander("Detail"):
                        for s, d in row['Sensor Detail'].items():
                            c = "red" if d['state'] == 'Critical' else "orange" if d['state'] == 'Warning' else "green"
                            icon = "🔴" if d['state'] == 'Critical' else "🟠" if d['state'] == 'Warning' else "🟢"
                            st.markdown(f"{icon} {s}: :{c}[{d['val']:.1f}]")