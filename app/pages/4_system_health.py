import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="System Health", page_icon="🏥", layout="wide")

st.title("🏥 System Health")

# 1. Privacy Budget
st.subheader("Privacy Budget Status")
epsilon_spent = 1.5
epsilon_total = 10.0
st.progress(epsilon_spent / epsilon_total)
st.write(f"**ε Spent:** {epsilon_spent} / {epsilon_total} (Total Budget)")

st.markdown("---")

# 2. Per-Node Metrics
st.subheader("Edge Node Metrics")
nodes = ["edge-node-1", "edge-node-2", "edge-node-3"]

metrics_df = pd.DataFrame(
    {
        "Node": nodes,
        "CPU (%)": [np.random.randint(40, 80) for _ in nodes],
        "RAM (MB)": [np.random.randint(500, 1400) for _ in nodes],
        "Throughput (req/s)": [np.random.randint(5, 15) for _ in nodes],
        "Avg Latency (ms)": [np.random.randint(50, 150) for _ in nodes],
    }
)

st.dataframe(
    metrics_df.style.highlight_max(subset=["CPU (%)", "Avg Latency (ms)"], color="#ff4b4b"),
    use_container_width=True,
)

# 3. Time Series Charts
col1, col2 = st.columns(2)

# Generate mock time series
times = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="1min")

ts_data = []
for node in nodes:
    base_lat = np.random.randint(80, 120)
    latencies = base_lat + np.random.normal(0, 10, 60)
    throughputs = np.random.normal(10, 2, 60)
    for i, t in enumerate(times):
        ts_data.append(
            {"Time": t, "Node": node, "Latency (ms)": latencies[i], "Throughput": throughputs[i]}
        )

ts_df = pd.DataFrame(ts_data)

with col1:
    fig_lat = px.line(ts_df, x="Time", y="Latency (ms)", color="Node", title="Latency (Last Hour)")
    st.plotly_chart(fig_lat, use_container_width=True)

with col2:
    fig_tp = px.line(ts_df, x="Time", y="Throughput", color="Node", title="Throughput (Last Hour)")
    st.plotly_chart(fig_tp, use_container_width=True)

st.markdown("---")
st.subheader("Qdrant Vector Database")
col_q1, col_q2, col_q3 = st.columns(3)
col_q1.metric("Collection Size", f"{np.random.randint(100000, 500000):,} vectors")
col_q2.metric("Query Latency", "12 ms")
col_q3.metric("Index Status", "Optimized", delta="Healthy")
