# backend_api.py (Backend Server) - versi perbaikan SHAP

# --- 1. SETTING VISUALISASI (WAJIB PALING ATAS) ---
import matplotlib
matplotlib.use("Agg")  # Mode tanpa layar (untuk server/API)
import matplotlib.pyplot as plt

# --- 2. IMPORT LIBRARY ---
import os
import io
import json
import base64
import traceback
from threading import Lock

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

# --- KONFIGURASI APP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def path(name):
    return os.path.join(BASE_DIR, name)

app = Flask(__name__)
CORS(app)
plot_lock = Lock()

# Kunci API dibaca dari environment variable, JANGAN ditulis di kode.
#   Windows (cmd):  set GEMINI_API_KEY=isi_kunci_baru
#   Linux/Mac:      export GEMINI_API_KEY=isi_kunci_baru
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDDOQ7JT6J1g2TzMgfX3Q4YqKKo_oSxjnI")
client = None
gemini_ready = False
if GEMINI_KEY:
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        gemini_ready = True
        print("✅ Gemini Client siap.")
    except Exception as e:
        print(f"❌ Gagal inisialisasi Gemini: {e}")
else:
    print("⚠️ GEMINI_API_KEY belum diset. Saran AI memakai fallback manual.")

# --- 3. LOAD MODEL (sama dengan pipeline di backend_anomali.ipynb) ---
COLS_TO_SCALE = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
COLS_OHE = ["Type_H", "Type_L", "Type_M"]
FEATS_OCSVM = COLS_TO_SCALE + COLS_OHE
FEATS_XGB = FEATS_OCSVM + ["Anomaly_Score"]

print("⚙️ Memuat model...")
model_ocsvm = joblib.load(path("model_ocsvm.pkl"))
scaler_ocsvm = joblib.load(path("scaler.pkl"))
model_hybrid = joblib.load(path("hybrid_xgb_model.pkl"))
scaler_xgb = joblib.load(path("hybrid_xgb_scaler.pkl"))
with open(path("thresholds.json")) as f:
    _th = json.load(f)
T_WARN = _th["Twarn"]
db_mesin = pd.read_csv(path("data_hasil_model.csv"))
explainer_xgb = shap.TreeExplainer(model_hybrid)
print("✅ Backend siap!")


# --- 4. FUNGSI SHAP ---
def make_shap_plot(x_scaled, x_display):
    """
    x_scaled  : array (1, n_fitur) yang dimasukkan ke model (sudah di-scale)
    x_display : Series nilai asli (sebelum scale) supaya label grafik mudah dibaca
    Mengembalikan gambar PNG dalam base64.
    """
    sv = explainer_xgb.shap_values(x_scaled)
    if isinstance(sv, list):          # kasus model mengembalikan list per kelas
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (1, n_fitur, n_kelas)
        sv = sv[:, :, -1]
    sv = sv[0]                        # WAJIB 1 dimensi untuk waterfall

    base = np.atleast_1d(explainer_xgb.expected_value)
    base = float(base[-1])

    exp = shap.Explanation(
        values=sv,
        base_values=base,
        data=x_display.values.astype(float),
        feature_names=list(x_display.index),
    )

    with plot_lock:
        plt.close("all")
        shap.plots.waterfall(exp, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close("all")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- 5. ENDPOINT ANALISIS ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.json or {}
        p_id = req.get("product_id")

        row = db_mesin[db_mesin["Product ID"] == p_id]
        if row.empty:
            return jsonify({"status": "error",
                            "message": f"Product ID '{p_id}' tidak ditemukan."}), 404
        row = row.iloc[0]

        type_str = str(row.get("Type", "L"))
        air = float(row["Air temperature [K]"])
        proc = float(row["Process temperature [K]"])
        rpm = float(row["Rotational speed [rpm]"])
        torque = float(row["Torque [Nm]"])
        wear = float(row["Tool wear [min]"])

        # Logika fisika
        temp_diff = proc - air
        power_w = 2 * np.pi * rpm / 60 * torque
        strain_load = torque * wear

        failure_reasons = []
        if power_w < 3500 or power_w > 9000:
            failure_reasons.append(f"Power Failure (Aktual: {int(power_w)} W)")
        if temp_diff < 8.6 and rpm < 1380:
            failure_reasons.append("Heat Dissipation Failure")
        if strain_load > 11000:
            failure_reasons.append(f"Overstrain (Beban: {int(strain_load)})")
        if wear >= 200:
            failure_reasons.append(f"Tool Wear Failure (Aus: {int(wear)} min)")

        df = pd.DataFrame({
            "Air temperature [K]": [air],
            "Process temperature [K]": [proc],
            "Rotational speed [rpm]": [rpm],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [wear],
            "Type_H": [1 if type_str == "H" else 0],
            "Type_L": [1 if type_str == "L" else 0],
            "Type_M": [1 if type_str == "M" else 0],
        })

        # 1) OCSVM -> anomaly score
        X_o = df[FEATS_OCSVM].copy()
        X_o[COLS_TO_SCALE] = scaler_ocsvm.transform(X_o[COLS_TO_SCALE])
        anomaly_score = float(model_ocsvm.decision_function(X_o)[0])

        # 2) XGBoost hybrid
        df["Anomaly_Score"] = anomaly_score
        X_xgb = df[FEATS_XGB].copy()
        X_xgb_scaled = scaler_xgb.transform(X_xgb)
        pred_proba = float(model_hybrid.predict_proba(X_xgb_scaled)[0][1])

        # Status
        if failure_reasons or pred_proba > 0.8:
            status = "Critical"
        elif anomaly_score < T_WARN or pred_proba > 0.5:
            status = "Warning"
        else:
            status = "Normal"

        # SHAP (dibuat untuk semua status supaya grafik selalu tampil)
        shap_b64, shap_error = "", ""
        try:
            shap_b64 = make_shap_plot(X_xgb_scaled, X_xgb.iloc[0])
        except Exception as e:
            shap_error = str(e)
            print("SHAP Error:", e)
            traceback.print_exc()

        # Saran tindakan
        advice = "Mesin beroperasi dalam batas normal."
        if status != "Normal":
            context = ("\n".join(f"- {x}" for x in failure_reasons)
                       if failure_reasons else "- Pola data anomali (ML)")
            advice = ""
            if gemini_ready:
                try:
                    prompt = (
                        f"Anda asisten teknisi pabrik. Mesin {p_id} berstatus {status}.\n"
                        f"Data:\n{context}\n"
                        f"Tipe {type_str}, Air {air} K, Proses {proc} K, {rpm} rpm, "
                        f"Torque {torque} Nm, Tool wear {wear} min.\n"
                        "Jawab dua bagian: 1. Akar Masalah  2. Instruksi Tindakan (1-2 langkah tegas)."
                    )
                    r = client.models.generate_content(
                        model="gemini-2.5-flash", contents=[prompt])
                    advice = r.text.strip()
                except Exception as e:
                    print("Gemini Error:", e)
            if not advice:
                tips = []
                s = str(failure_reasons)
                if "Power" in s: tips.append("Cek tegangan listrik dan inverter motor.")
                if "Heat" in s: tips.append("Periksa sistem pendingin dan sirkulasi udara.")
                if "Overstrain" in s: tips.append("Kurangi beban torsi, indikasi overload.")
                if "Tool Wear" in s: tips.append("Mata pahat habis, ganti tool baru.")
                if not tips: tips.append("Lakukan inspeksi visual pada bearing dan poros.")
                advice = "**Saran Manual (AI Offline):**\n" + "\n".join(f"- {t}" for t in tips)

        return jsonify({
            "status": status,
            "probability": f"{pred_proba * 100:.1f}",
            "ocsvm_score": f"{anomaly_score:.2f}",
            "ai_analysis": advice,
            "shap_image": shap_b64,
            "shap_error": shap_error,
            "machine_info": {
                "Type": type_str,
                "Air temperature [K]": air,
                "Process temperature [K]": proc,
                "Rotational speed [rpm]": int(rpm),
                "Torque [Nm]": torque,
                "Tool wear [min]": int(wear),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# --- 6. ENDPOINT CHATBOT ---
@app.route("/chat-bot", methods=["POST"])
def chat_bot():
    if not gemini_ready:
        return jsonify({"reply": "AI belum terhubung. Cek GEMINI_API_KEY.", "status": "Error"}), 500
    try:
        msg = (request.json or {}).get("message", "")
        r = client.models.generate_content(model="gemini-2.5-flash", contents=[msg])
        return jsonify({"reply": r.text.strip(), "status": "Normal"})
    except Exception:
        return jsonify({"reply": "Terjadi kesalahan pada server AI.", "status": "Error"}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)