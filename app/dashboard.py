import time

import streamlit as st

st.set_page_config(
    page_title="Kizuna Privacy Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_header():
    st.sidebar.title("Kizuna Dashboard")
    st.sidebar.markdown("---")

    # System Status Badge
    # Mocking status for layout purposes
    system_status = "Online"
    status_color = "green" if system_status == "Online" else "red"
    st.sidebar.markdown(f"**System Status**: :{status_color}[{system_status}]")

    # Node count and last event timestamp
    st.sidebar.markdown("**Active Edge Nodes**: 3")

    # Mock last event
    last_event_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    st.sidebar.markdown(f"**Last Event**: {last_event_time}")

    st.sidebar.markdown("---")
    st.sidebar.info("Use the navigation above to switch pages.")


def main():
    st.title("🛡️ Kizuna Multimodal Privacy Engine")

    st.markdown(
        """
    Welcome to the Kizuna Dashboard. This interface provides real-time monitoring
    and historical analysis of edge-processed privacy events.

    Please select a page from the sidebar to continue:
    - **Live Monitor**: Watch events stream in real time.
    - **Anomaly History**: Search and filter historical privacy alerts.
    - **Vector Explorer**: Visually inspect the embedding space clusters.
    - **System Health**: Monitor edge node CPU, Memory, and Throughput.
    """
    )

    render_header()


if __name__ == "__main__":
    main()
