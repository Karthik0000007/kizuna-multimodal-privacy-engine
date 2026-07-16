import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np

from src.anomaly.detector import AnomalyOrchestrator
from src.database.qdrant_client import QdrantStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("central-node")

# Global instances (simplified for HTTP server)
vector_store = None
orchestrator = None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

    pass


class CentralNodeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/ingest":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                payload = json.loads(post_data.decode("utf-8"))
                vector = np.array(payload["vector"], dtype=np.float32)
                metadata = payload["metadata"]

                # Insert into vector store
                vector_store.insert(vector, metadata)

                # Run anomaly detection
                event = orchestrator.process(vector, metadata["source_node_id"])

                if event and event.is_anomaly:
                    logger.warning(
                        f"🚨 PRIVACY EVENT DETECTED: {event.anomaly_type} from {event.source_node_id} (conf: {event.confidence:.2f})"
                    )

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "processed"}')

            except Exception as e:
                logger.error(f"Error processing ingest: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Only log non-200 or errors to keep it clean
        if args[1] != "200":
            logger.info(
                "%s - - [%s] %s"
                % (self.address_string(), self.log_date_time_string(), format % args)
            )


def main():
    global vector_store, orchestrator

    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
    bind_host = os.environ.get("BIND_HOST", "0.0.0.0")
    bind_port = int(os.environ.get("BIND_PORT", 8001))

    logger.info("Initializing central aggregation node...")

    # Init DB
    try:
        vector_store = QdrantStore(
            host=qdrant_host, port=qdrant_port, collection_name="kizuna_edge_sim"
        )
        vector_store.create_collection(dimension=512)
        logger.info("Connected to Qdrant")
    except Exception as e:
        logger.error(f"Could not connect to Qdrant: {e}")
        # In a real app we might retry, but let's fallback to FAISS for testing if Qdrant isn't there
        from src.database.faiss_store import FAISSStore

        vector_store = FAISSStore("data/central_faiss.bin", "data/central_meta.json")
        vector_store.create_collection(dimension=512)
        logger.info("Fell back to local FAISS store")

    orchestrator = AnomalyOrchestrator(vector_store)

    server = ThreadedHTTPServer((bind_host, bind_port), CentralNodeHandler)
    logger.info(f"Central node listening on {bind_host}:{bind_port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down central node")
        server.server_close()


if __name__ == "__main__":
    main()
