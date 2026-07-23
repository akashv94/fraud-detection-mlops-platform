import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Fraud Detection MLOps", layout="wide")
st.title("Fraud Detection MLOps Platform")
st.caption("Streamlit client -> FastAPI -> XGBoost")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.write("API health:", health)
except Exception:
    st.warning(f"Could not connect to API at {API_URL}")

st.subheader("Transaction prediction")

with st.form("transaction"):
    values = {}
    values["Time"] = st.number_input("Time", value=0.0)
    values["Amount"] = st.number_input("Amount", min_value=0.0, value=100.0)

    with st.expander("V1-V28 features"):
        for i in range(1, 29):
            values[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {"Time": values["Time"]}
    for i in range(1, 29):
        payload[f"V{i}"] = values[f"V{i}"]
    payload["Amount"] = values["Amount"]

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.ok:
            result = response.json()
            st.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
            st.write("Prediction:", result["label"].upper())
            st.json(result)
        else:
            st.error(response.text)
    except Exception as exc:
        st.error(str(exc))
