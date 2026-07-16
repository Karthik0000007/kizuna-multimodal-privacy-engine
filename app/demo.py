import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Kizuna Privacy Engine Demo",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Kizuna Privacy Engine - Interactive Demo")
st.markdown("Experience edge-based, privacy-preserving multimodal anomaly detection.")

# --- Sidebar Configuration ---
st.sidebar.header("Simulation Settings")
scenario = st.sidebar.selectbox(
    "Scenario to Simulate",
    [
        "Normal Operations",
        "Fall Risk (Camera 1)",
        "Crowd Congestion (Corridor A)",
        "Wandering (Exit B)",
    ],
)

st.sidebar.markdown("### Modality Fusion")
use_video = st.sidebar.checkbox("Video Encoder (Vision)", value=True)
use_audio = st.sidebar.checkbox("Audio Encoder (Sound)", value=True)
use_env = st.sidebar.checkbox("Env Sensors (Temp/Hum)", value=True)

st.sidebar.markdown("### Differential Privacy Budget")
epsilon = st.sidebar.slider("DP Epsilon (ε)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
st.sidebar.caption("Lower ε = Higher Privacy, More Noise")

st.sidebar.markdown("---")
if st.sidebar.button("Start Live Stream"):
    st.session_state["streaming"] = True
if st.sidebar.button("Stop Live Stream"):
    st.session_state["streaming"] = False

# --- Main App State ---
if "streaming" not in st.session_state:
    st.session_state["streaming"] = False
if "history" not in st.session_state:
    st.session_state["history"] = []

# Layout
col1, col2, col3 = st.columns(3)
metric_latency = col1.empty()
metric_budget = col2.empty()
metric_status = col3.empty()

plot_placeholder = st.empty()
log_placeholder = st.empty()


def generate_embedding(scenario_name, noise_scale):
    """Generate a synthetic 2D projection of an embedding for visualization"""
    if scenario_name == "Normal Operations":
        base = np.array([0.0, 0.0])
    elif "Fall Risk" in scenario_name:
        base = np.array([5.0, 5.0])
    elif "Crowd" in scenario_name:
        base = np.array([-5.0, 5.0])
    elif "Wandering" in scenario_name:
        base = np.array([5.0, -5.0])

    # Add DP Noise
    noise = np.random.laplace(0, 1.0 / epsilon, size=2)
    return base + noise


# --- Streaming Loop ---
if st.session_state["streaming"]:
    budget_spent = 0.0
    for _i in range(100):
        if not st.session_state["streaming"]:
            break

        start_time = time.perf_counter()

        # 1. Generate new point
        point = generate_embedding(scenario, epsilon)

        # 2. Determine Anomaly Status
        distance = np.linalg.norm(point)
        is_anomaly = distance > 3.0

        # Update metrics
        budget_spent += epsilon
        latency = (time.perf_counter() - start_time) * 1000 + np.random.uniform(
            15, 25
        )  # Mock edge latency

        st.session_state["history"].append(
            {
                "x": point[0],
                "y": point[1],
                "type": "Anomaly" if is_anomaly else "Normal",
                "time": time.strftime("%H:%M:%S"),
            }
        )

        # Keep last 50 points
        if len(st.session_state["history"]) > 50:
            st.session_state["history"].pop(0)

        # Draw metrics
        metric_latency.metric("Edge Processing Latency", f"{latency:.1f} ms", "-0.2 ms")
        metric_budget.metric("Privacy Budget Used", f"{budget_spent:.1f} ε", f"+{epsilon} ε")

        status_text = "🚨 ANOMALY DETECTED" if is_anomaly else "✅ NORMAL"
        metric_status.metric("System Status", status_text)

        # Draw Plot
        df = pd.DataFrame(st.session_state["history"])
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="type",
            title="Live 2D PCA of Unified Multimodal Embeddings",
            color_discrete_map={"Normal": "green", "Anomaly": "red"},
            range_x=[-10, 10],
            range_y=[-10, 10],
        )
        plot_placeholder.plotly_chart(fig, use_container_width=True)

        # Logs
        log_df = df.tail(5).iloc[::-1]
        log_placeholder.dataframe(log_df, use_container_width=True)

        time.sleep(0.5)
else:
    st.info("👈 Click 'Start Live Stream' in the sidebar to begin the simulation.")
