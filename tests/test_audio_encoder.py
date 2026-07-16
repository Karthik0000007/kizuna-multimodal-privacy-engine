"""Unit tests for audio encoder."""

# Import the audio encoder
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.export_audio_onnx import SimpleAudioEncoder, export_audio_model
from src.engine.audio_encoder import AudioEncoder


@pytest.fixture
def temp_model_path():
    """Create a temporary audio model for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_audio_model.onnx"

        # Export a simple audio model
        export_audio_model(
            output_path=model_path,
            n_mels=128,
            embedding_dim=512,
            sample_duration=1.0,
            sample_rate=16000,
            hop_length=512,
        )

        yield model_path


class TestAudioEncoder:
    """Test suite for AudioEncoder."""

    def test_init_success(self, temp_model_path):
        """Test successful initialization."""
        encoder = AudioEncoder(
            model_path=temp_model_path,
            target_sample_rate=16000,
            n_mels=128,
        )

        assert encoder.model_path == temp_model_path
        assert encoder.embedding_dim == 512
        assert encoder.target_sample_rate == 16000
        assert encoder.n_mels == 128

    def test_init_file_not_found(self):
        """Test initialization with non-existent model file."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            AudioEncoder(model_path="nonexistent_model.onnx")

    def test_encode_output_shape(self, temp_model_path):
        """Test that encode returns correct output shape."""
        encoder = AudioEncoder(model_path=temp_model_path)

        # Create test audio (1 second at 16kHz)
        audio_chunk = np.random.randn(16000).astype(np.float32)

        embedding = encoder.encode(audio_chunk)

        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32

    def test_encode_deterministic(self, temp_model_path):
        """Test that encode produces deterministic output for same input."""
        encoder = AudioEncoder(model_path=temp_model_path)

        # Same input
        audio_chunk = np.random.randn(16000).astype(np.float32)

        embedding1 = encoder.encode(audio_chunk)
        embedding2 = encoder.encode(audio_chunk)

        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)

    def test_encode_l2_normalized(self, temp_model_path):
        """Test that encoded embedding is L2-normalized."""
        encoder = AudioEncoder(model_path=temp_model_path)

        audio_chunk = np.random.randn(16000).astype(np.float32)
        embedding = encoder.encode(audio_chunk)

        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5), f"Expected norm=1.0, got {norm}"

    def test_encode_invalid_shape(self, temp_model_path):
        """Test encode with invalid audio shape."""
        encoder = AudioEncoder(model_path=temp_model_path)

        # 2D array instead of 1D
        audio_chunk = np.random.randn(2, 16000).astype(np.float32)

        with pytest.raises(ValueError, match="Expected audio_chunk shape"):
            encoder.encode(audio_chunk)

    def test_encode_invalid_dtype(self, temp_model_path):
        """Test encode with invalid audio dtype."""
        encoder = AudioEncoder(model_path=temp_model_path)

        # int16 instead of float32
        audio_chunk = np.random.randint(-32768, 32767, 16000, dtype=np.int16)

        with pytest.raises(ValueError, match="Expected audio_chunk dtype"):
            encoder.encode(audio_chunk)

    def test_encode_padded_audio(self, temp_model_path):
        """Test encode with audio shorter than expected length (should be padded)."""
        encoder = AudioEncoder(model_path=temp_model_path, chunk_duration=1.0)

        # Audio shorter than 1 second
        audio_chunk = np.random.randn(8000).astype(np.float32)

        # Should not raise, should pad internally
        embedding = encoder.encode(audio_chunk)

        assert embedding.shape == (512,)

    def test_encode_trimmed_audio(self, temp_model_path):
        """Test encode with audio longer than expected length (should be trimmed)."""
        encoder = AudioEncoder(model_path=temp_model_path, chunk_duration=1.0)

        # Audio longer than 1 second
        audio_chunk = np.random.randn(24000).astype(np.float32)

        # Should not raise, should trim internally
        embedding = encoder.encode(audio_chunk)

        assert embedding.shape == (512,)

    def test_encode_batch_output_shape(self, temp_model_path):
        """Test that encode_batch returns correct output shape."""
        encoder = AudioEncoder(model_path=temp_model_path)

        # Create batch of test audio
        batch_size = 4
        audio_chunks = np.random.randn(batch_size, 16000).astype(np.float32)

        embeddings = encoder.encode_batch(audio_chunks)

        assert embeddings.shape == (batch_size, 512)
        assert embeddings.dtype == np.float32

    def test_encode_batch_l2_normalized(self, temp_model_path):
        """Test that batch embeddings are L2-normalized per row."""
        encoder = AudioEncoder(model_path=temp_model_path)

        audio_chunks = np.random.randn(3, 16000).astype(np.float32)
        embeddings = encoder.encode_batch(audio_chunks)

        # Check L2 norm for each row
        for i in range(embeddings.shape[0]):
            norm = np.linalg.norm(embeddings[i])
            assert np.isclose(norm, 1.0, atol=1e-5), f"Row {i}: Expected norm=1.0, got {norm}"

    def test_encode_batch_invalid_shape(self, temp_model_path):
        """Test encode_batch with invalid audio shape."""
        encoder = AudioEncoder(model_path=temp_model_path)

        # 1D array instead of 2D
        audio_chunks = np.random.randn(16000).astype(np.float32)

        with pytest.raises(ValueError, match="Expected audio_chunks shape"):
            encoder.encode_batch(audio_chunks)

    def test_encode_batch_invalid_dtype(self, temp_model_path):
        """Test encode_batch with invalid audio dtype."""
        encoder = AudioEncoder(model_path=temp_model_path)

        # int16 instead of float32
        audio_chunks = np.random.randint(-32768, 32767, (2, 16000), dtype=np.int16)

        with pytest.raises(ValueError, match="Expected audio_chunks dtype"):
            encoder.encode_batch(audio_chunks)

    def test_warm_up(self, temp_model_path):
        """Test warm_up runs successfully and returns latency."""
        encoder = AudioEncoder(model_path=temp_model_path)

        avg_latency = encoder.warm_up(num_iterations=5)

        assert isinstance(avg_latency, (float, np.floating))
        assert avg_latency > 0

    def test_get_embedding_dim(self, temp_model_path):
        """Test get_embedding_dim returns correct dimension."""
        encoder = AudioEncoder(model_path=temp_model_path)

        dim = encoder.get_embedding_dim()

        assert dim == 512

    def test_different_embeddings_for_different_inputs(self, temp_model_path):
        """Test that different audio inputs produce different embeddings."""
        encoder = AudioEncoder(model_path=temp_model_path)

        # Two different audio chunks
        audio1 = np.random.randn(16000).astype(np.float32)
        audio2 = np.random.randn(16000).astype(np.float32)

        embedding1 = encoder.encode(audio1)
        embedding2 = encoder.encode(audio2)

        # Embeddings should be different
        cosine_sim = np.dot(embedding1, embedding2)  # Already L2-normalized
        assert cosine_sim < 0.99, "Different inputs should produce different embeddings"

    def test_similar_embeddings_for_similar_inputs(self, temp_model_path):
        """Test that similar audio inputs produce similar embeddings."""
        encoder = AudioEncoder(model_path=temp_model_path)

        # Create two similar audio chunks (same + noise)
        audio1 = np.random.randn(16000).astype(np.float32)
        audio2 = audio1 + 0.01 * np.random.randn(16000).astype(np.float32)

        embedding1 = encoder.encode(audio1)
        embedding2 = encoder.encode(audio2)

        # Embeddings should be similar
        cosine_sim = np.dot(embedding1, embedding2)  # Already L2-normalized
        assert cosine_sim > 0.8, "Similar inputs should produce similar embeddings"


class TestSimpleAudioEncoder:
    """Test suite for SimpleAudioEncoder PyTorch model."""

    def test_forward_output_shape(self):
        """Test forward pass produces correct output shape."""
        model = SimpleAudioEncoder(n_mels=128, embedding_dim=512)
        model.eval()

        # Input: (batch_size, n_mels, time_frames)
        x = torch.randn(2, 128, 31)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 512)

    def test_forward_l2_normalized(self):
        """Test forward pass produces L2-normalized embeddings."""
        model = SimpleAudioEncoder(n_mels=128, embedding_dim=512)
        model.eval()

        x = torch.randn(1, 128, 31)

        with torch.no_grad():
            output = model(x)

        norm = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-5)

    def test_forward_with_4d_input(self):
        """Test forward pass with 4D input (batch, channels, n_mels, time)."""
        model = SimpleAudioEncoder(n_mels=128, embedding_dim=512)
        model.eval()

        # Input with channel dimension
        x = torch.randn(2, 1, 128, 31)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 512)

    def test_forward_deterministic(self):
        """Test forward pass is deterministic."""
        model = SimpleAudioEncoder(n_mels=128, embedding_dim=512, dropout=0.0)
        model.eval()

        x = torch.randn(1, 128, 31)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)


def test_export_audio_model():
    """Test export_audio_model creates valid ONNX model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_audio.onnx"

        export_audio_model(
            output_path=output_path,
            n_mels=128,
            embedding_dim=512,
            sample_duration=1.0,
            sample_rate=16000,
            hop_length=512,
        )

        # Check file exists
        assert output_path.exists()

        # Check file is valid ONNX
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        # Check model metadata
        assert len(onnx_model.graph.input) == 1
        assert len(onnx_model.graph.output) == 1
        assert onnx_model.graph.input[0].name == "mel_spectrogram"
        assert onnx_model.graph.output[0].name == "embedding"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
