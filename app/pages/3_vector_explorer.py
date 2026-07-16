import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Vector Explorer", page_icon="🌌", layout="wide")

st.title("🌌 Vector Explorer")

st.markdown(
    """
Explore the 512-dimensional embedding space projected into 2D using simulated t-SNE.
Clusters indicate behavioral similarities across modalities.
"""
)


@st.cache_data
def generate_tsne_data():
    np.random.seed(42)

    # Generate mock clusters
    n_normal = 500
    n_fall = 50
    n_wander = 50

    # Normal cluster (centered at 0,0)
    normal_x = np.random.normal(0, 1, n_normal)
    normal_y = np.random.normal(0, 1, n_normal)

    # Fall risk cluster (centered at 5,5)
    fall_x = np.random.normal(5, 0.5, n_fall)
    fall_y = np.random.normal(5, 0.5, n_fall)

    # Wandering cluster (centered at -4, 4)
    wander_x = np.random.normal(-4, 0.8, n_wander)
    wander_y = np.random.normal(4, 0.8, n_wander)

    x = np.concatenate([normal_x, fall_x, wander_x])
    y = np.concatenate([normal_y, fall_y, wander_y])
    labels = ["normal"] * n_normal + ["fall_risk"] * n_fall + ["wandering"] * n_wander
    nodes = [f"edge-node-{np.random.randint(1, 4)}" for _ in range(len(x))]

    df = pd.DataFrame(
        {
            "tsne_1": x,
            "tsne_2": y,
            "Event Type": labels,
            "Source Node": nodes,
            "Size": [10 if label == "normal" else 20 for label in labels],
        }
    )
    return df


df = generate_tsne_data()

color_map = {"normal": "blue", "fall_risk": "red", "wandering": "orange"}

fig = px.scatter(
    df,
    x="tsne_1",
    y="tsne_2",
    color="Event Type",
    hover_data=["Source Node"],
    color_discrete_map=color_map,
    size="Size",
    opacity=0.7,
    title="2D Projection of Modality Embeddings",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Nearest Neighbor Search")
st.info(
    "Interactive 'Find Similar' would allow clicking a point on the chart to query Qdrant for K-nearest neighbors. (Mocked for UI Layout)"
)

if st.button("Find Similar to Last Event"):
    st.write("Found 3 similar events:")
    st.json(
        [
            {"id": "evt-123", "distance": 0.05, "type": "fall_risk"},
            {"id": "evt-456", "distance": 0.12, "type": "fall_risk"},
            {"id": "evt-789", "distance": 0.18, "type": "normal"},
        ]
    )
