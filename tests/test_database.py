import os
import shutil
import time

import numpy as np
import pytest

from src.database.faiss_store import FAISSStore
from src.database.qdrant_client import QdrantStore


@pytest.fixture
def sample_metadata():
    return {
        "timestamp": time.time(),
        "source_node_id": "node-1",
        "modalities_fused": ["video", "audio"],
        "event_type": "fall",
        "dp_epsilon": 1.0,
    }


@pytest.fixture
def sample_vector():
    vec = np.random.rand(512)
    return vec / np.linalg.norm(vec)


class TestDatabaseStore:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # We test FAISS locally without needing Qdrant container running
        self.faiss_dir = "tests/temp_faiss_data"
        os.makedirs(self.faiss_dir, exist_ok=True)
        self.faiss_store = FAISSStore(
            index_path=f"{self.faiss_dir}/index.bin",
            metadata_path=f"{self.faiss_dir}/metadata.json",
        )
        self.faiss_store.create_collection(dimension=512)

        yield

        # Teardown
        self.faiss_store.delete_collection()
        if os.path.exists(self.faiss_dir):
            shutil.rmtree(self.faiss_dir)

    def test_faiss_insert_and_search(self, sample_vector, sample_metadata):
        # Insert
        point_id = self.faiss_store.insert(sample_vector, sample_metadata)
        assert point_id is not None

        # Search exact match
        results = self.faiss_store.search(sample_vector, top_k=1)
        assert len(results) == 1
        assert results[0].id == point_id
        assert results[0].score > 0.99  # Cosine similarity should be approx 1
        assert results[0].payload["source_node_id"] == "node-1"

    def test_faiss_metadata_filtering(self, sample_vector, sample_metadata):
        self.faiss_store.insert(sample_vector, sample_metadata)

        # Different metadata
        meta2 = sample_metadata.copy()
        meta2["source_node_id"] = "node-2"
        meta2["event_type"] = "wandering"
        vec2 = np.random.rand(512)
        vec2 = vec2 / np.linalg.norm(vec2)

        self.faiss_store.insert(vec2, meta2)

        # Search with filter for node-2
        filters = {"source_node_id": "node-2"}
        results = self.faiss_store.search(sample_vector, top_k=5, filters=filters)

        assert len(results) == 1
        assert results[0].payload["source_node_id"] == "node-2"
        assert results[0].payload["event_type"] == "wandering"

        # Search with filter for specific event
        filters_event = {"event_type": "fall"}
        results_event = self.faiss_store.search(vec2, top_k=5, filters=filters_event)

        assert len(results_event) == 1
        assert results_event[0].payload["event_type"] == "fall"

    def test_faiss_persistence(self, sample_vector, sample_metadata):
        point_id = self.faiss_store.insert(sample_vector, sample_metadata)
        self.faiss_store._save()

        # Create a new instance pointing to same paths
        new_store = FAISSStore(
            index_path=f"{self.faiss_dir}/index.bin",
            metadata_path=f"{self.faiss_dir}/metadata.json",
        )

        assert new_store.current_count == 1
        assert new_store.dimension == 512

        results = new_store.search(sample_vector, top_k=1)
        assert len(results) == 1
        assert results[0].id == point_id


@pytest.mark.integration
class TestQdrantStoreIntegration:
    """
    These tests require a running Qdrant instance.
    They will be skipped automatically if Qdrant is unreachable.
    """

    def qdrant_is_available(self):
        try:
            client = QdrantStore(host="localhost", port=6333, collection_name="test_health")
            client.client.get_collections()
            return True
        except Exception:
            return False

    @pytest.fixture(autouse=True)
    def setup_qdrant(self):
        if not self.qdrant_is_available():
            pytest.skip("Qdrant is not running locally. Skipping integration tests.")

        self.qdrant = QdrantStore(host="localhost", port=6333, collection_name="test_integration")
        self.qdrant.create_collection(dimension=512)

        yield

        self.qdrant.delete_collection()

    def test_qdrant_insert_and_search(self, sample_vector, sample_metadata):
        point_id = self.qdrant.insert(sample_vector, sample_metadata)
        assert point_id is not None

        # Give Qdrant a moment to index
        time.sleep(0.5)

        results = self.qdrant.search(sample_vector, top_k=1)
        assert len(results) == 1
        assert results[0].id == point_id
        assert results[0].score > 0.99
        assert results[0].payload["source_node_id"] == "node-1"

    def test_qdrant_metadata_filtering(self, sample_vector, sample_metadata):
        self.qdrant.insert(sample_vector, sample_metadata)

        meta2 = sample_metadata.copy()
        meta2["source_node_id"] = "node-2"
        vec2 = np.random.rand(512)
        vec2 = vec2 / np.linalg.norm(vec2)

        self.qdrant.insert(vec2, meta2)

        time.sleep(0.5)

        filters = {"source_node_id": "node-2"}
        results = self.qdrant.search(sample_vector, top_k=5, filters=filters)

        assert len(results) == 1
        assert results[0].payload["source_node_id"] == "node-2"
