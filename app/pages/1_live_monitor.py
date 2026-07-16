import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Live Monitor", page_icon="📡", layout="wide")

st.title("📡 Live Monitor")

# Auto-refresh mechanism (using a simple counter in session_state and time.sleep loop, or user button for now)
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# To make Streamlit auto-refresh naturally without external components,
# we can use st.rerun() after a sleep, but that blocks interaction.
# We will just add a manual refresh button, or simulate the live data feed on render.

cols = st.columns(3)

# 1. Per-node status cards
nodes = ["edge-node-1", "edge-node-2", "edge-node-3"]
for i, node in enumerate(nodes):
    with cols[i]:
        st.subheader(node)
        st.write("Status: :green[Online]")
        # Mock sparkline
        sparkline_data = np.random.normal(50, 10, size=100)
        st.line_chart(sparkline_data, height=100)
        st.write(f"Payloads processed: {np.random.randint(1000, 5000)}")
        st.write(f"Last seen: {datetime.now().strftime('%H:%M:%S')}")

st.markdown("---")

# 2. Current anomaly alert banner (Simulated if active)
is_alert_active = np.random.choice([True, False], p=[0.3, 0.7])
if is_alert_active:
    st.error("🚨 **ACTIVE ANOMALY DETECTED**")
    st.warning(
        f"**Event Type**: elderly_fall | **Confidence**: {np.random.uniform(0.8, 0.99):.2f} | **Source**: edge-node-{np.random.randint(1, 4)}"
    )
else:
    st.success("✅ **System Normal**: No active anomalies.")

st.markdown("---")

# 3. Real-time event feed (scrolling log)
st.subheader("Real-Time Event Feed (Last 50)")

# Generate dummy event log
events = []
for i in range(50):
    events.append(
        {
            "Timestamp": (datetime.now() - timedelta(seconds=i * 2)).strftime("%H:%M:%S"),
            "Node": f"edge-node-{np.random.randint(1, 4)}",
            "Event Type": np.random.choice(
                ["normal", "normal", "normal", "fall_risk", "unusual_sound"]
            ),
            "Latency (ms)": f"{np.random.uniform(10, 150):.1f}",
        }
    )

df = pd.DataFrame(events)


# Apply simple styling
def color_anomaly(val):
    color = "red" if val != "normal" else "green"
    return f"color: {color}"


styled_df = df.style.map(color_anomaly, subset=["Event Type"])
st.dataframe(styled_df, use_container_width=True, height=400)

if st.button("Refresh Feed"):
    st.rerun()
