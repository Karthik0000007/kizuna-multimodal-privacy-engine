import json
import os
import time
from datetime import datetime

import streamlit as st
import yaml

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings")

# ---------- Current Configuration (Read-Only) ----------
st.subheader("Current Configuration")

# Load the default config YAML if it exists, otherwise show a placeholder
config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "default.yaml")
config_path = os.path.normpath(config_path)

config_data = {}
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}

if config_data:
    st.code(yaml.dump(config_data, default_flow_style=False, sort_keys=False), language="yaml")
else:
    st.info("No configuration file found. Using system defaults.")
    config_data = {
        "ingestion": {"frame_rate": 15, "resolution": [224, 224], "audio_sample_rate": 16000},
        "embedding": {"vision_model": "models/vision/model_int8.onnx", "dimension": 512},
        "privacy": {"mechanism": "laplace", "epsilon": 1.0, "delta": 1e-5, "budget_ceiling": 10.0},
        "anomaly": {
            "knn_threshold": 0.8,
            "density_threshold": 1.5,
            "cluster_threshold": 0.7,
            "ensemble_min_votes": 2,
        },
        "database": {
            "backend": "qdrant",
            "host": "localhost",
            "port": 6333,
            "collection": "kizuna_embeddings",
        },
        "dashboard": {"refresh_rate_hz": 1, "alert_verbosity": "medium"},
        "edge_simulation": {"num_nodes": 3, "cpu_per_node": 2, "ram_per_node_gb": 2},
    }
    st.code(yaml.dump(config_data, default_flow_style=False, sort_keys=False), language="yaml")

st.markdown("---")

# ---------- Runtime Adjustments ----------
st.subheader("Runtime Adjustments")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Anomaly Detection Thresholds**")
    knn_thresh = st.slider("KNN Threshold", 0.0, 1.0, 0.8, 0.05, key="knn")
    density_thresh = st.slider("Density Threshold", 0.0, 3.0, 1.5, 0.1, key="density")
    cluster_thresh = st.slider("Cluster Threshold", 0.0, 1.0, 0.7, 0.05, key="cluster")

with col2:
    st.markdown("**Privacy Settings**")
    dp_epsilon = st.slider("DP Epsilon (ε)", 0.01, 10.0, 1.0, 0.01, key="epsilon")
    alert_verbosity = st.selectbox(
        "Alert Verbosity", ["low", "medium", "high"], index=1, key="verbosity"
    )

if st.button("Apply Changes"):
    st.success(
        "Settings updated successfully! (Note: In production, these would be persisted to config and hot-reloaded.)"
    )
    st.json(
        {
            "anomaly": {
                "knn_threshold": knn_thresh,
                "density_threshold": density_thresh,
                "cluster_threshold": cluster_thresh,
            },
            "privacy": {"epsilon": dp_epsilon},
            "dashboard": {"alert_verbosity": alert_verbosity},
        }
    )

st.markdown("---")

# ---------- Download Buttons ----------
st.subheader("Export")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    # Download current config as YAML
    config_yaml_str = yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    st.download_button(
        label="📥 Download Config (YAML)",
        data=config_yaml_str.encode("utf-8"),
        file_name="kizuna_config.yaml",
        mime="text/yaml",
    )

with col_dl2:
    # Generate a mock audit log for download
    audit_log = []
    for i in range(20):
        audit_log.append(
            {
                "timestamp": (datetime.now().timestamp() - i * 60),
                "event": "payload_processed" if i % 3 != 0 else "anomaly_detected",
                "node": f"edge-node-{(i % 3) + 1}",
                "epsilon_spent": round(0.1 * (i + 1), 2),
            }
        )

    audit_json = json.dumps(audit_log, indent=2)
    st.download_button(
        label="📥 Download Audit Log (JSON)",
        data=audit_json.encode("utf-8"),
        file_name="kizuna_audit_log.json",
        mime="application/json",
    )
