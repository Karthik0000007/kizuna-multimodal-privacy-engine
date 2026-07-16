import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("edge-node")


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging for health checks


def run_health_server():
    server = HTTPServer(("0.0.0.0", 8000), HealthHandler)
    logger.info("Health server listening on port 8000")
    server.serve_forever()


def main():
    node_id = os.environ.get("NODE_ID", "local-edge")
    central_url = os.environ.get("CENTRAL_NODE_URL", "http://localhost:8001")

    logger.info(f"Starting edge node {node_id}")

    # Start health server in background
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Graceful shutdown requested")
        running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Main simulation loop
    while running:
        try:
            start_time = time.perf_counter_ns()

            # 1. Ingest (Simulated)
            # 2. Embed (Simulated)
            embedding = np.random.rand(512)
            embedding /= np.linalg.norm(embedding)

            # 3. Privatize (Simulated DP Noise)
            noise = np.random.laplace(0, 0.01, size=512)
            private_embedding = embedding + noise

            # 4. Transmit
            payload = {
                "vector": private_embedding.tolist(),
                "metadata": {
                    "timestamp": time.time(),
                    "source_node_id": node_id,
                    "modalities_fused": ["video"],
                    "event_type": "normal",
                    "dp_epsilon": 1.0,
                },
            }

            req = urllib.request.Request(
                f"{central_url}/ingest",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status != 200:
                        logger.warning(f"Central node returned status {response.status}")
            except Exception as e:
                logger.error(f"Failed to transmit to central node: {e}")

            end_time = time.perf_counter_ns()
            latency_ms = (end_time - start_time) / 1_000_000

            logger.info(f"Processed payload in {latency_ms:.2f}ms")

            # Wait before next payload
            time.sleep(1.0)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(1.0)

    logger.info("Edge node stopped.")


if __name__ == "__main__":
    main()
