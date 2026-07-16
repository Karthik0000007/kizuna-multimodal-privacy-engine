from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Anomaly History", page_icon="📅", layout="wide")

st.title("📅 Anomaly History")


# Generate mock historical data
@st.cache_data
def load_historical_data():
    np.random.seed(42)
    dates = [datetime.now() - timedelta(hours=i) for i in range(24 * 7)]

    events = []
    for d in dates:
        # random number of anomalies in this hour
        num_anomalies = np.random.poisson(lam=1.5)
        for _ in range(num_anomalies):
            events.append(
                {
                    "Timestamp": d.strftime("%Y-%m-%d %H:%M:%S"),
                    "Node": f"edge-node-{np.random.randint(1, 4)}",
                    "Event Type": np.random.choice(
                        ["fall_risk", "wandering", "unusual_sound", "congestion_alert"],
                        p=[0.4, 0.3, 0.2, 0.1],
                    ),
                    "Confidence": round(np.random.uniform(0.7, 0.99), 2),
                    "Modalities": ", ".join(
                        np.random.choice(
                            ["video", "audio", "sensor"],
                            size=np.random.randint(1, 4),
                            replace=False,
                        )
                    ),
                }
            )
    return pd.DataFrame(events)


df = load_historical_data()

# Filters
st.sidebar.header("Filters")
event_types = st.sidebar.multiselect(
    "Event Type", options=df["Event Type"].unique(), default=df["Event Type"].unique()
)
nodes = st.sidebar.multiselect(
    "Source Node", options=df["Node"].unique(), default=df["Node"].unique()
)
min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.7)

# Apply filters
filtered_df = df[
    (df["Event Type"].isin(event_types)) & (df["Node"].isin(nodes)) & (df["Confidence"] >= min_conf)
]

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Anomalies Over Time (Last 7 Days)")
    # Group by date for bar chart
    filtered_df["Date"] = pd.to_datetime(filtered_df["Timestamp"]).dt.date
    daily_counts = filtered_df.groupby("Date").size().reset_index(name="Count")
    fig_bar = px.bar(daily_counts, x="Date", y="Count", color_discrete_sequence=["#ff4b4b"])
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Anomaly Type Distribution")
    type_counts = filtered_df["Event Type"].value_counts().reset_index()
    type_counts.columns = ["Event Type", "Count"]
    fig_pie = px.pie(type_counts, values="Count", names="Event Type", hole=0.3)
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

st.subheader("Event Records")
st.dataframe(filtered_df.drop(columns=["Date"]), use_container_width=True)

# CSV Export
csv = filtered_df.drop(columns=["Date"]).to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name="anomaly_history.csv",
    mime="text/csv",
)
