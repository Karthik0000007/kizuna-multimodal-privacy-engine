import os
import shutil
import sys

# To make this self-contained for Hugging Face Spaces, we can simply run the demo script.
# Hugging Face Spaces automatically runs `app.py` if Streamlit SDK is chosen.

# We will just copy the contents of the demo file or import it.
# Easiest way in a single file space is to execute the demo script directly.

if __name__ == "__main__":
    # If this is run by Streamlit, it will execute the following file
    demo_path = os.path.join(os.path.dirname(__file__), "..", "app", "demo.py")
    if os.path.exists(demo_path):
        with open(demo_path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code)
    else:
        # Fallback for standalone HF Space structure
        st.error("Demo file not found.")
