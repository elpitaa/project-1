import streamlit as st
from joblib import load
import numpy as np
import pandas as pd

# ===============================
# Load Model & Scaler
# ===============================
# model and scaler were saved with joblib (sklearn). Use joblib.load to read them.
# Muat model dan scaler yang disimpan dengan joblib
model = load("knn_model.pkl")
scaler = load("minmax_scaler.pkl")

# Tentukan berapa fitur yang diharapkan
expected_n_features = None
if hasattr(scaler, "n_features_in_"):
    expected_n_features = int(scaler.n_features_in_)
elif hasattr(model, "n_features_in_"):
    expected_n_features = int(model.n_features_in_)


# ===============================
# Judul Aplikasi
# ===============================
st.title("Prediksi Nilai NO₂ Satu Hari Kedepan dengan KNN Regression")
st.markdown(
    """
    Aplikasi ini memprediksi konsentrasi **NO₂ (Nitrogen Dioksida)** satu hari ke depan
    berdasarkan data input fitur lingkungan.
    
    Model yang digunakan: **K-Nearest Neighbors (KNN) Regression**
    """
)

# ===============================
# Input Fitur
# ===============================
st.subheader("Masukkan Nilai Fitur")

# Kamu bisa ubah daftar fitur sesuai dengan data training kamu
# Default daftar fitur yang ditampilkan ke user. Jika jumlah fitur yang diharapkan
# berbeda, kita akan menyesuaikan nama fitur menjadi generic (Feature 1, ...).
default_feature_names = [
    "PM10", "SO2", "CO", "O3", "Temperature", "Humidity", "WindSpeed"
]

if expected_n_features is None:
    feature_names = default_feature_names
else:
    if expected_n_features == len(default_feature_names):
        feature_names = default_feature_names
    else:
        # Buat nama generic agar UI menyesuaikan dengan scaler/model yang ada
        feature_names = [f"Feature_{i+1}" for i in range(expected_n_features)]
        st.warning(
            f"Model saat ini mengharapkan {expected_n_features} fitur.\n"
            "Gunakan urutan fitur yang sama saat model dilatih.\n"
            "Jika Anda ingin memakai nama fitur yang spesifik, edit kode aplikasi ini."
        )
        # Beri informasi internal scaler untuk membantu pemetaan fitur
        try:
            st.caption(f"Info scaler: n_features_in_={getattr(scaler,'n_features_in_',None)}, min_={getattr(scaler,'min_',None)}, scale_={getattr(scaler,'scale_',None)}")
        except Exception:
            pass

# Siapkan nilai default yang masuk akal: gunakan mid-point pada skala (0.5) jika scaler tersedia
default_inputs = None
try:
    if hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
        # transform inverse: X = (X_scaled - min_) / scale_
        mid_scaled = np.full(int(expected_n_features or len(feature_names)), 0.5)
        default_inputs = ((mid_scaled - scaler.min_) / scaler.scale_).astype(float)
except Exception:
    default_inputs = None

inputs = []
col1, col2 = st.columns(2)
for i, name in enumerate(feature_names):
    with (col1 if i % 2 == 0 else col2):
        default_val = float(default_inputs[i]) if (default_inputs is not None and i < len(default_inputs)) else 0.0
        value = st.number_input(f"{name}", value=default_val)
        inputs.append(value)

# ===============================
# Prediksi
# ===============================
if st.button("Prediksi NO₂"):
    # Ubah input menjadi array numpy
    data = np.array(inputs).reshape(1, -1)
    try:
        # Normalisasi dengan scaler
        data_scaled = scaler.transform(data)

        # Prediksi dengan model KNN
        pred = model.predict(data_scaled)

        # Tampilkan hasil dan informasi debug kecil
        # Tampilkan prediksi dengan presisi lebih tinggi dan juga versi yang dibulatkan
        pred_val = float(pred[0])
        st.success(f"Prediksi Konsentrasi NO₂ Besok: {pred_val:.6f} µg/m³")
        st.caption(f"(dibulatkan ke 2 desimal: {pred_val:.2f} µg/m³)")
        with st.expander("Detail input & transform"):
            st.write({
                'raw_input': data.tolist(),
                'scaled_input': data_scaled.tolist(),
                'prediction': pred.tolist()
            })
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")

# ===============================
# Catatan Tambahan
# ===============================
st.markdown("---")
st.caption("Dibuat menggunakan Streamlit | Model: KNN Regression")
