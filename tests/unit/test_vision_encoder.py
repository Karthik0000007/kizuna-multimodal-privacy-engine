"""Unit tests for vision encoder."""

import numpy as np
import pytest

from src.engine.vision_encoder import VisionEncoder

# Note: These tests require an ONNX model to be present
# Run: python scripts/export_vision_onnx.py --model simple
# to create the model before running tests


class TestVisionEncoder:
    """Tests for VisionEncoder class."""

    @pytest.fixture
    def model_path(self, tmp_path):
        """Create a temporary model path for testing."""
        # Try to use actual model if available
        from pathlib import Path

        actual_model = Path("models/vision/model.onnx")
        if actual_model.exists():
            return actual_model

        # For CI/CD, we'll skip tests that require the model
        pytest.skip("ONNX model not found. Run: python scripts/export_vision_onnx.py")

    @pytest.mark.requires_models
    def test_init_valid_model(self, model_path):
        """Test initialization with valid model."""
        encoder = VisionEncoder(model_path=model_path)

        assert encoder.model_path.exists()
        assert encoder.embedding_dim > 0
        assert encoder.input_size == (224, 224)

    def test_init_invalid_model_path(self, tmp_path):
        """Test initialization with invalid model path."""
        invalid_path = tmp_path / "nonexistent.onnx"

        with pytest.raises(FileNotFoundError):
            VisionEncoder(model_path=invalid_path)

    @pytest.mark.requires_models
    def test_encode_valid_frame(self, model_path):
        """Test encoding valid frame."""
        encoder = VisionEncoder(model_path=model_path)

        # Create test frame
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Encode
        embedding = encoder.encode(frame)

        assert embedding.shape == (encoder.embedding_dim,)
        assert embedding.dtype == np.float32
        assert 0.99 <= np.linalg.norm(embedding) <= 1.01  # L2 normalized

    @pytest.mark.requires_models
    def test_encode_invalid_shape(self, model_path):
        """Test encoding with invalid frame shape."""
        encoder = VisionEncoder(model_path=model_path)

        # Invalid shape (grayscale)
        frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

        with pytest.raises(ValueError, match="Expected frame shape"):
            encoder.encode(frame)

    @pytest.mark.requires_models
    def test_encode_invalid_dtype(self, model_path):
        """Test encoding with invalid frame dtype."""
        encoder = VisionEncoder(model_path=model_path)

        # Invalid dtype (float32 instead of uint8)
        frame = np.random.rand(480, 640, 3).astype(np.float32)

        with pytest.raises(ValueError, match="Expected frame dtype"):
            encoder.encode(frame)

    @pytest.mark.requires_models
    def test_encode_deterministic(self, model_path):
        """Test encoding is deterministic for same input."""
        encoder = VisionEncoder(model_path=model_path)

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Encode same frame twice
        embedding1 = encoder.encode(frame)
        embedding2 = encoder.encode(frame)

        # Should be identical
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)

    @pytest.mark.requires_models
    def test_encode_batch(self, model_path):
        """Test batch encoding."""
        encoder = VisionEncoder(model_path=model_path)

        batch_size = 4
        frames = np.random.randint(0, 256, (batch_size, 480, 640, 3), dtype=np.uint8)

        # Encode batch
        embeddings = encoder.encode_batch(frames)

        assert embeddings.shape == (batch_size, encoder.embedding_dim)
        assert embeddings.dtype == np.float32

        # Each embedding should be L2 normalized
        for i in range(batch_size):
            norm = np.linalg.norm(embeddings[i])
            assert 0.99 <= norm <= 1.01

    @pytest.mark.requires_models
    def test_encode_batch_vs_single(self, model_path):
        """Test batch encoding matches individual encoding."""
        encoder = VisionEncoder(model_path=model_path)

        # Create test frames
        frames = np.random.randint(0, 256, (3, 480, 640, 3), dtype=np.uint8)

        # Batch encoding
        batch_embeddings = encoder.encode_batch(frames)

        # Individual encoding
        single_embeddings = np.stack([encoder.encode(frame) for frame in frames])

        # Should be very close (minor numerical differences allowed)
        np.testing.assert_array_almost_equal(batch_embeddings, single_embeddings, decimal=4)

    @pytest.mark.requires_models
    def test_warm_up(self, model_path):
        """Test model warm-up."""
        encoder = VisionEncoder(model_path=model_path)

        num_iterations = 5
        avg_latency = encoder.warm_up(num_iterations=num_iterations)

        assert avg_latency > 0  # Should take some time
        assert isinstance(avg_latency, float)

    @pytest.mark.requires_models
    def test_get_embedding_dim(self, model_path):
        """Test getting embedding dimension."""
        encoder = VisionEncoder(model_path=model_path)

        dim = encoder.get_embedding_dim()

        assert isinstance(dim, int)
        assert dim > 0

    @pytest.mark.requires_models
    def test_preprocess_resize(self, model_path):
        """Test preprocessing resizes correctly."""
        encoder = VisionEncoder(model_path=model_path)

        # Large frame
        frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

        # Preprocess
        processed = encoder._preprocess(frame)

        # Should be resized to input_size
        if encoder.channel_order == "CHW":
            assert processed.shape == (1, 3, *encoder.input_size)
        else:  # HWC
            assert processed.shape == (1, *encoder.input_size, 3)

    @pytest.mark.requires_models
    def test_preprocess_normalization(self, model_path):
        """Test preprocessing normalization."""
        encoder = VisionEncoder(model_path=model_path)

        # Create frame with known values
        frame = np.full((224, 224, 3), 128, dtype=np.uint8)  # Mid-gray

        # Preprocess
        processed = encoder._preprocess(frame)

        # Values should be normalized (not in [0, 255] range)
        assert processed.min() < 1.0
        assert processed.max() > -1.0

    def test_normalize_l2(self):
        """Test L2 normalization."""
        # Random vector
        vector = np.random.randn(512).astype(np.float32)

        # Normalize
        normalized = VisionEncoder._normalize_l2(vector)

        # Should have norm ~1
        norm = np.linalg.norm(normalized)
        assert 0.99 <= norm <= 1.01

    def test_normalize_l2_zero_vector(self):
        """Test L2 normalization of zero vector."""
        # Zero vector
        vector = np.zeros(512, dtype=np.float32)

        # Normalize (should handle gracefully)
        normalized = VisionEncoder._normalize_l2(vector)

        # Should still be zero
        np.testing.assert_array_equal(normalized, vector)

    @pytest.mark.requires_models
    def test_different_input_sizes(self, model_path):
        """Test encoding frames of different sizes."""
        encoder = VisionEncoder(model_path=model_path)

        # Different sized frames
        sizes = [(224, 224), (480, 640), (720, 1280), (1080, 1920)]

        for h, w in sizes:
            frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

            # Should handle any size (will be resized internally)
            embedding = encoder.encode(frame)

            assert embedding.shape == (encoder.embedding_dim,)
            assert 0.99 <= np.linalg.norm(embedding) <= 1.01

    @pytest.mark.requires_models
    def test_embedding_similarity(self, model_path):
        """Test embeddings of similar images are similar."""
        encoder = VisionEncoder(model_path=model_path)

        # Create base frame
        base_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Create slightly modified frame (add small noise)
        noise = np.random.randint(-5, 6, base_frame.shape, dtype=np.int16)
        similar_frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Encode both
        embedding1 = encoder.encode(base_frame)
        embedding2 = encoder.encode(similar_frame)

        # Compute cosine similarity
        cosine_sim = np.dot(embedding1, embedding2)

        # Should be very similar (cosine similarity > 0.9)
        assert cosine_sim > 0.9

    @pytest.mark.requires_models
    def test_embedding_dissimilarity(self, model_path):
        """Test embeddings of different images are different."""
        encoder = VisionEncoder(model_path=model_path)

        # Create two completely different frames
        frame1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Encode both
        embedding1 = encoder.encode(frame1)
        embedding2 = encoder.encode(frame2)

        # Compute cosine similarity
        cosine_sim = np.dot(embedding1, embedding2)

        # Should not be identical (cosine similarity < 0.99)
        assert cosine_sim < 0.99


class TestVisionEncoderEdgeCases:
    """Edge case tests for VisionEncoder."""

    @pytest.mark.requires_models
    def test_very_small_frame(self, tmp_path):
        """Test encoding very small frame."""
        # Skip if model not available
        from pathlib import Path

        model_path = Path("models/vision/model.onnx")
        if not model_path.exists():
            pytest.skip("ONNX model not found")

        encoder = VisionEncoder(model_path=model_path)

        # Very small frame (will be upscaled)
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        # Should still work
        embedding = encoder.encode(frame)
        assert embedding.shape == (encoder.embedding_dim,)

    @pytest.mark.requires_models
    def test_very_large_frame(self, tmp_path):
        """Test encoding very large frame."""
        from pathlib import Path

        model_path = Path("models/vision/model.onnx")
        if not model_path.exists():
            pytest.skip("ONNX model not found")

        encoder = VisionEncoder(model_path=model_path)

        # Very large frame (will be downscaled)
        frame = np.random.randint(0, 256, (2160, 3840, 3), dtype=np.uint8)

        # Should still work
        embedding = encoder.encode(frame)
        assert embedding.shape == (encoder.embedding_dim,)

    @pytest.mark.requires_models
    def test_black_frame(self, tmp_path):
        """Test encoding completely black frame."""
        from pathlib import Path

        model_path = Path("models/vision/model.onnx")
        if not model_path.exists():
            pytest.skip("ONNX model not found")

        encoder = VisionEncoder(model_path=model_path)

        # Black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Should produce valid embedding
        embedding = encoder.encode(frame)
        assert embedding.shape == (encoder.embedding_dim,)
        assert 0.99 <= np.linalg.norm(embedding) <= 1.01

    @pytest.mark.requires_models
    def test_white_frame(self, tmp_path):
        """Test encoding completely white frame."""
        from pathlib import Path

        model_path = Path("models/vision/model.onnx")
        if not model_path.exists():
            pytest.skip("ONNX model not found")

        encoder = VisionEncoder(model_path=model_path)

        # White frame
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)

        # Should produce valid embedding
        embedding = encoder.encode(frame)
        assert embedding.shape == (encoder.embedding_dim,)
        assert 0.99 <= np.linalg.norm(embedding) <= 1.01
