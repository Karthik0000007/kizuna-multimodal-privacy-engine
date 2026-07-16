import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def generate_demo_gif(output_path="docs/demo.gif"):
    """Generates a synthetic animated GIF representing the demo output."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title("Kizuna Demo: Multimodal Embedding Space")

    # Normal state cluster
    (normal_points,) = ax.plot([], [], "go", label="Normal (Ambient)")
    # Anomaly cluster
    (anomaly_points,) = ax.plot([], [], "ro", label="Anomaly Detected")

    ax.legend(loc="upper right")

    normal_x, normal_y = [], []
    anomaly_x, anomaly_y = [], []

    def init():
        normal_points.set_data([], [])
        anomaly_points.set_data([], [])
        return normal_points, anomaly_points

    def update(frame):
        # First 20 frames: Normal
        # Frame 20-50: Anomaly
        # Frame 50+: Normal

        if frame < 20 or frame > 50:
            # Normal
            nx = np.random.normal(0, 1.0)
            ny = np.random.normal(0, 1.0)
            normal_x.append(nx)
            normal_y.append(ny)
        else:
            # Anomaly (e.g. Fall Risk at 5,5)
            ax = np.random.normal(5, 1.5)
            ay = np.random.normal(5, 1.5)
            anomaly_x.append(ax)
            anomaly_y.append(ay)

        # Keep only last 20 points for trailing effect
        nx_tail = normal_x[-20:]
        ny_tail = normal_y[-20:]
        ax_tail = anomaly_x[-20:]
        ay_tail = anomaly_y[-20:]

        normal_points.set_data(nx_tail, ny_tail)
        anomaly_points.set_data(ax_tail, ay_tail)

        return normal_points, anomaly_points

    ani = animation.FuncAnimation(fig, update, frames=80, init_func=init, blit=True, interval=100)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Require pillow for saving gif
        ani.save(output_path, writer="pillow", fps=10)
        print(f"Successfully generated {output_path}")
    except Exception as e:
        print(f"Could not generate GIF (is Pillow installed?): {e}")


if __name__ == "__main__":
    generate_demo_gif()
