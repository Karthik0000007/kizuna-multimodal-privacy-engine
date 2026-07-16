"""Configuration management for Kizuna Privacy Engine.

This module provides configuration loading, validation, and management
with support for YAML files, environment variables, and runtime overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class SystemConfig(BaseModel):
    """System-wide configuration."""

    name: str = "kizuna-central"
    version: str = "0.1.0"
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_dir: str = "logs"
    correlation_id_enabled: bool = True


class VideoConfig(BaseModel):
    """Video ingestion configuration."""

    enabled: bool = True
    fps: int = Field(default=15, ge=1, le=60)
    resolution: List[int] = Field(default=[320, 320])
    source_type: str = Field(default="simulator", pattern="^(simulator|file|camera)$")
    source_path: Optional[str] = None

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: List[int]) -> List[int]:
        """Validate resolution is [width, height] with positive values."""
        if len(v) != 2:
            raise ValueError("Resolution must be [width, height]")
        if any(dim <= 0 for dim in v):
            raise ValueError("Resolution dimensions must be positive")
        return v


class AudioConfig(BaseModel):
    """Audio ingestion configuration."""

    enabled: bool = True
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    chunk_duration: float = Field(default=1.0, gt=0.0, le=10.0)
    source_type: str = Field(default="simulator", pattern="^(simulator|file|microphone)$")
    source_path: Optional[str] = None


class EnvironmentalConfig(BaseModel):
    """Environmental sensor configuration."""

    enabled: bool = True
    sensors: List[str] = Field(
        default=["temperature", "humidity", "motion", "light", "air_quality"]
    )
    polling_rate: float = Field(default=1.0, gt=0.0)


class TemporalAlignmentConfig(BaseModel):
    """Temporal alignment configuration."""

    enabled: bool = True
    window_size: float = Field(default=1.0, gt=0.0)
    jitter_tolerance: float = Field(default=0.05, ge=0.0)
    buffer_size: int = Field(default=100, ge=1)


class BackpressureConfig(BaseModel):
    """Backpressure configuration."""

    enabled: bool = True
    max_queue_size: int = Field(default=1000, ge=1)
    policy: str = Field(default="newest_wins", pattern="^(newest_wins|oldest_wins|drop_random)$")


class IngestionConfig(BaseModel):
    """Data ingestion configuration."""

    enabled: bool = True
    modalities: List[str] = Field(default=["video", "audio", "environmental"])
    video: VideoConfig = Field(default_factory=VideoConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    environmental: EnvironmentalConfig = Field(default_factory=EnvironmentalConfig)
    temporal_alignment: TemporalAlignmentConfig = Field(default_factory=TemporalAlignmentConfig)
    backpressure: BackpressureConfig = Field(default_factory=BackpressureConfig)


class VisionModelConfig(BaseModel):
    """Vision model configuration."""

    model_path: str = "models/vision/model_int8.onnx"
    preprocessing: Dict[str, Any] = Field(
        default_factory=lambda: {
            "resize": [224, 224],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "channel_order": "CHW",
        }
    )


class AudioModelConfig(BaseModel):
    """Audio model configuration."""

    model_path: str = "models/audio/model_int8.onnx"
    preprocessing: Dict[str, Any] = Field(
        default_factory=lambda: {
            "target_sample_rate": 16000,
            "n_mels": 128,
            "n_fft": 2048,
            "hop_length": 512,
        }
    )


class SensorModelConfig(BaseModel):
    """Sensor model configuration."""

    model_path: str = "models/sensor/model_int8.onnx"
    normalization_ranges: Dict[str, List[float]] = Field(
        default_factory=lambda: {
            "temperature": [15.0, 40.0],
            "humidity": [20.0, 90.0],
            "motion": [0.0, 1.0],
            "light": [0.0, 1000.0],
            "air_quality": [0.0, 500.0],
        }
    )


class FusionConfig(BaseModel):
    """Multimodal fusion configuration."""

    strategy: str = Field(default="late_fusion", pattern="^(late_fusion|early_fusion)$")
    projection_head_path: str = "models/fusion/projection.onnx"
    handle_missing: bool = True


class RuntimeConfig(BaseModel):
    """ONNX Runtime configuration."""

    execution_provider: str = Field(
        default="CPUExecutionProvider", pattern="^(CPUExecutionProvider|CUDAExecutionProvider)$"
    )
    intra_op_num_threads: int = Field(default=2, ge=1)
    inter_op_num_threads: int = Field(default=1, ge=1)
    graph_optimization_level: str = "ORT_ENABLE_ALL"


class EmbeddingConfig(BaseModel):
    """Embedding engine configuration."""

    enabled: bool = True
    output_dimension: int = Field(default=512, ge=128, le=2048)
    normalization: str = Field(default="l2", pattern="^(l2|none)$")
    vision: VisionModelConfig = Field(default_factory=VisionModelConfig)
    audio: AudioModelConfig = Field(default_factory=AudioModelConfig)
    sensor: SensorModelConfig = Field(default_factory=SensorModelConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


class DifferentialPrivacyConfig(BaseModel):
    """Differential privacy configuration."""

    enabled: bool = True
    mechanism: str = Field(default="laplace", pattern="^(laplace|gaussian)$")
    epsilon: float = Field(default=1.0, gt=0.0)
    delta: float = Field(default=1e-5, gt=0.0, lt=1.0)
    sensitivity: float = Field(default=2.0, gt=0.0)


class MemoryWipingConfig(BaseModel):
    """Memory wiping configuration."""

    enabled: bool = True
    method: str = Field(default="native", pattern="^(native|python)$")
    overwrite_passes: int = Field(default=1, ge=1, le=10)
    verify: bool = True


class BudgetTrackingConfig(BaseModel):
    """Privacy budget tracking configuration."""

    enabled: bool = True
    total_budget: float = Field(default=10.0, gt=0.0)
    alert_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    composition: str = Field(default="sequential", pattern="^(sequential|parallel)$")
    persistence_path: str = "logs/privacy_budget.json"


class AuditLoggingConfig(BaseModel):
    """Audit logging configuration."""

    enabled: bool = True
    log_all_events: bool = True
    log_path: str = "logs/privacy_audit.log"


class PrivacyConfig(BaseModel):
    """Privacy layer configuration."""

    enabled: bool = True
    differential_privacy: DifferentialPrivacyConfig = Field(
        default_factory=DifferentialPrivacyConfig
    )
    memory_wiping: MemoryWipingConfig = Field(default_factory=MemoryWipingConfig)
    budget_tracking: BudgetTrackingConfig = Field(default_factory=BudgetTrackingConfig)
    audit_logging: AuditLoggingConfig = Field(default_factory=AuditLoggingConfig)


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""

    host: str = "localhost"
    port: int = Field(default=6333, ge=1, le=65535)
    collection_name: str = "kizuna_embeddings"
    vector_dimension: int = Field(default=512, ge=128)
    distance_metric: str = Field(default="cosine", pattern="^(cosine|euclidean|dot)$")
    hnsw_config: Dict[str, int] = Field(
        default_factory=lambda: {"m": 16, "ef_construct": 100, "ef_search": 50}
    )
    connection: Dict[str, Any] = Field(
        default_factory=lambda: {"timeout": 30, "retry_attempts": 3, "retry_backoff": 2.0}
    )
    persistence: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": True, "storage_path": "qdrant_storage"}
    )
    retention: Dict[str, Any] = Field(default_factory=lambda: {"enabled": True, "ttl_days": 30})


class FAISSConfig(BaseModel):
    """FAISS vector database configuration."""

    index_type: str = Field(default="IndexFlatIP", pattern="^(IndexFlatIP|IndexFlatL2)$")
    persistence_path: str = "data/faiss_index.bin"
    max_vectors: int = Field(default=1000000, ge=1)
    save_interval: int = Field(default=1000, ge=1)


class DatabaseConfig(BaseModel):
    """Vector database configuration."""

    backend: str = Field(default="qdrant", pattern="^(qdrant|faiss)$")
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)


class AnomalyDetectorConfig(BaseModel):
    """Individual anomaly detector configuration."""

    enabled: bool = True
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)


class AnomalyConfig(BaseModel):
    """Anomaly detection configuration."""

    enabled: bool = True
    detectors: List[str] = Field(default=["knn", "density", "cluster"])
    knn: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": True, "k": 10, "threshold": 0.7, "weight": 1.0}
    )
    density: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "algorithm": "lof",
            "n_neighbors": 20,
            "threshold": 1.5,
            "weight": 1.0,
        }
    )
    cluster: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "baseline_clusters": ["normal_walk", "sitting_still", "crowd_flow"],
            "cluster_threshold": 0.6,
            "weight": 1.0,
        }
    )
    ensemble: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "voting_strategy": "majority",
            "min_votes": 2,
            "confidence_aggregation": "weighted_average",
        }
    )
    classification: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "event_types": [
                "fall_risk",
                "wandering",
                "congestion_alert",
                "unusual_sound",
                "environmental_anomaly",
            ],
            "top_k": 3,
        }
    )


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = Field(default=8501, ge=1, le=65535)
    title: str = "Kizuna Privacy Engine Dashboard"
    auto_refresh_interval: float = Field(default=1.0, gt=0.0)
    max_events_display: int = Field(default=50, ge=1)
    pages: List[str] = Field(
        default=[
            "live_monitor",
            "anomaly_history",
            "vector_explorer",
            "system_health",
            "settings",
        ]
    )
    vector_explorer: Dict[str, Any] = Field(
        default_factory=lambda: {
            "projection_method": "umap",
            "projection_dim": 2,
            "recompute_interval": 300,
            "cache_projections": True,
        }
    )


class EdgeSimulationConfig(BaseModel):
    """Edge simulation configuration."""

    enabled: bool = False
    node_id: str = "central-node"
    resource_constraints: Dict[str, float] = Field(
        default_factory=lambda: {"cpu_cores": 4, "memory_gb": 4.0}
    )
    transmission: Dict[str, Any] = Field(
        default_factory=lambda: {
            "protocol": "grpc",
            "target_host": "localhost",
            "target_port": 50051,
            "batch_size": 10,
            "buffer_size": 1000,
            "retry_attempts": 5,
        }
    )


class TelemetryConfig(BaseModel):
    """Telemetry configuration."""

    enabled: bool = True
    metrics: List[str] = Field(
        default=["latency", "throughput", "memory_usage", "cpu_usage", "anomaly_count"]
    )
    export: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "backend": "prometheus", "port": 9090}
    )


class KizunaConfig(BaseModel):
    """Root configuration for Kizuna Privacy Engine."""

    system: SystemConfig = Field(default_factory=SystemConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    edge_simulation: EdgeSimulationConfig = Field(default_factory=EdgeSimulationConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


class ConfigManager:
    """Configuration manager with loading, validation, and merging.

    Supports:
    - YAML file loading
    - Environment variable overrides (KIZUNA_* prefix)
    - Schema validation via Pydantic
    - Configuration merging with precedence
    - Default config generation
    """

    ENV_PREFIX = "KIZUNA_"

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file. If None, uses default.yaml
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[KizunaConfig] = None

    def load(self, config_path: Optional[Union[str, Path]] = None) -> KizunaConfig:
        """Load and validate configuration.

        Args:
            config_path: Override config path if provided

        Returns:
            Validated KizunaConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            pydantic.ValidationError: If validation fails
        """
        if config_path:
            self.config_path = Path(config_path)

        # Load from file or use defaults
        if self.config_path and self.config_path.exists():
            config_dict = self._load_yaml(self.config_path)
        else:
            # Generate default config if missing
            config_dict = {}
            if self.config_path:
                self._generate_default_config(self.config_path)

        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)

        # Validate and create config object
        self._config = KizunaConfig(**config_dict)

        return self._config

    def validate(self, config_dict: Dict[str, Any]) -> KizunaConfig:
        """Validate configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Validated KizunaConfig object

        Raises:
            pydantic.ValidationError: If validation fails
        """
        return KizunaConfig(**config_dict)

    def merge(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries (deep merge).

        Args:
            base_config: Base configuration
            override_config: Override configuration (takes precedence)

        Returns:
            Merged configuration dictionary
        """
        result = base_config.copy()

        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge(result[key], value)
            else:
                result[key] = value

        return result

    def get_config(self) -> KizunaConfig:
        """Get current loaded configuration.

        Returns:
            Current KizunaConfig object

        Raises:
            RuntimeError: If config hasn't been loaded yet
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        return self._config

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML configuration file.

        Args:
            path: Path to YAML file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.

        Environment variables should use format: KIZUNA_SECTION_KEY=value
        Example: KIZUNA_SYSTEM_LOG_LEVEL=DEBUG

        Args:
            config_dict: Base configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(self.ENV_PREFIX):
                continue

            # Parse environment variable key
            config_key = env_key[len(self.ENV_PREFIX) :].lower()
            keys = config_key.split("_")

            # Navigate to nested dict and set value
            current = config_dict
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = self._parse_env_value(env_value)

        return config_dict

    @staticmethod
    def _parse_env_value(value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Parsed value (str, int, float, or bool)
        """
        # Try boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Try numeric
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string
        return value

    @staticmethod
    def _generate_default_config(path: Path) -> None:
        """Generate default configuration file.

        Args:
            path: Path where to save default config
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        default_config = KizunaConfig()
        config_dict = default_config.model_dump()

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
