
import streamlit as st
import pandas as pd
import joblib
import time
from utils import preprocess, FEATURE_COLUMNS

@st.cache_resource
def load_models():
    return {
        "Local Outlier Factor": joblib.load("models/lof_model.pkl"),
        "One-Class SVM": joblib.load("models/ocsvm_model.pkl"),
        "Isolation Forest": joblib.load("models/isolation_forest_model.pkl"),
        "K-Means": joblib.load("models/kmeans_model.pkl"),
        "Gaussian Mixture": joblib.load("models/gmm_model.pkl"),
        "Elliptic Envelope": joblib.load("models/elliptic_model.pkl"),
    }

models = load_models()

st.title("🚨 Real-Time Network Anomaly Detection Dashboard")
st.markdown("Using **Snort + ML Models** for live threat prediction.")

log_path = "live_logs/snort_live.csv"
refresh_interval = st.sidebar.slider("Refresh every (seconds)", 1, 10, 3)

if not FEATURE_COLUMNS:
    st.error("Feature list is empty.")
    st.stop()

placeholder = st.empty()
last_seen = 0
while True:
    try:
        df = pd.read_csv(log_path)
        if len(df) <= last_seen:
            time.sleep(refresh_interval)
            st.experimental_rerun()

        new_data = df.iloc[last_seen:]
        last_seen = len(df)
        processed_data = preprocess(new_data)

        results = {}
        for name, model in models.items():
            preds = model.predict(processed_data)
            results[name] = ["🟢 Normal" if p != -1 else "🔴 Anomaly" for p in preds]

        display_df = new_data.copy()
        for model_name, pred_list in results.items():
            display_df[model_name] = pred_list

        placeholder.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        break

    time.sleep(refresh_interval)
    st.experimental_rerun()
