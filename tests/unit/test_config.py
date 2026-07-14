"""Unit tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import ConfigManager, KizunaConfig


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_load_default_config(self) -> None:
        """Test loading default configuration."""
        manager = ConfigManager()
        config = manager.load()

        assert isinstance(config, KizunaConfig)
        assert config.system.name == "kizuna-central"
        assert config.system.version == "0.1.0"
        assert config.system.log_level == "INFO"

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Test loading configuration from YAML file."""
        config_file = tmp_path / "test_config.yaml"
        test_config = {
            "system": {"name": "test-node", "log_level": "DEBUG"},
            "privacy": {"enabled": True},
        }

        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        manager = ConfigManager(config_file)
        config = manager.load()

        assert config.system.name == "test-node"
        assert config.system.log_level == "DEBUG"

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable overrides."""
        config_file = tmp_path / "test_config.yaml"
        test_config = {"system": {"log_level": "INFO"}}

        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        # Set environment variable
        monkeypatch.setenv("KIZUNA_SYSTEM_LOG_LEVEL", "ERROR")

        manager = ConfigManager(config_file)
        config = manager.load()

        assert config.system.log_level == "ERROR"

    def test_env_override_nested(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable override for nested config."""
        config_file = tmp_path / "test_config.yaml"
        test_config = {"privacy": {"differential_privacy": {"epsilon": 1.0}}}

        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        # Set nested environment variable
        monkeypatch.setenv("KIZUNA_PRIVACY_DIFFERENTIAL_PRIVACY_EPSILON", "5.0")

        manager = ConfigManager(config_file)
        config = manager.load()

        assert config.privacy.differential_privacy.epsilon == 5.0

    def test_validation_fps_range(self) -> None:
        """Test validation rejects invalid FPS values."""
        invalid_config = {"ingestion": {"video": {"fps": 100}}}  # Max is 60

        with pytest.raises(ValidationError) as exc_info:
            KizunaConfig(**invalid_config)

        assert "fps" in str(exc_info.value).lower()

    def test_validation_resolution_format(self) -> None:
        """Test validation rejects invalid resolution format."""
        invalid_config = {"ingestion": {"video": {"resolution": [640]}}}  # Must be [w, h]

        with pytest.raises(ValidationError) as exc_info:
            KizunaConfig(**invalid_config)

        assert "resolution" in str(exc_info.value).lower()

    def test_validation_negative_resolution(self) -> None:
        """Test validation rejects negative resolution values."""
        invalid_config = {"ingestion": {"video": {"resolution": [-1, 224]}}}

        with pytest.raises(ValidationError) as exc_info:
            KizunaConfig(**invalid_config)

        assert "resolution" in str(exc_info.value).lower()

    def test_validation_epsilon_positive(self) -> None:
        """Test validation requires positive epsilon."""
        invalid_config = {"privacy": {"differential_privacy": {"epsilon": 0.0}}}

        with pytest.raises(ValidationError) as exc_info:
            KizunaConfig(**invalid_config)

        assert "epsilon" in str(exc_info.value).lower()

    def test_validation_port_range(self) -> None:
        """Test validation of port number ranges."""
        invalid_config = {"database": {"qdrant": {"port": 70000}}}  # Max is 65535

        with pytest.raises(ValidationError) as exc_info:
            KizunaConfig(**invalid_config)

        assert "port" in str(exc_info.value).lower()

    def test_merge_configs(self) -> None:
        """Test merging two configuration dictionaries."""
        base = {"system": {"name": "base", "log_level": "INFO"}, "privacy": {"enabled": True}}
        override = {"system": {"log_level": "DEBUG"}, "anomaly": {"enabled": False}}

        manager = ConfigManager()
        merged = manager.merge(base, override)

        assert merged["system"]["name"] == "base"  # Preserved from base
        assert merged["system"]["log_level"] == "DEBUG"  # Overridden
        assert merged["privacy"]["enabled"] is True  # Preserved from base
        assert merged["anomaly"]["enabled"] is False  # Added from override

    def test_generate_default_config(self, tmp_path: Path) -> None:
        """Test generation of default configuration file."""
        config_file = tmp_path / "generated_config.yaml"

        manager = ConfigManager(config_file)
        config = manager.load()  # Should generate default since file doesn't exist

        assert config_file.exists()
        assert isinstance(config, KizunaConfig)

        # Verify generated file is valid YAML
        with open(config_file, "r") as f:
            loaded = yaml.safe_load(f)
            assert "system" in loaded
            assert "ingestion" in loaded
            assert "privacy" in loaded

    def test_get_config_before_load(self) -> None:
        """Test that get_config raises error if not loaded."""
        manager = ConfigManager()

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_config()

        assert "not loaded" in str(exc_info.value).lower()

    def test_get_config_after_load(self, tmp_path: Path) -> None:
        """Test get_config returns loaded configuration."""
        config_file = tmp_path / "test_config.yaml"
        test_config = {"system": {"name": "test"}}

        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        manager = ConfigManager(config_file)
        config1 = manager.load()
        config2 = manager.get_config()

        assert config1 is config2  # Same object

    def test_parse_env_value_boolean(self) -> None:
        """Test parsing boolean environment variables."""
        manager = ConfigManager()

        assert manager._parse_env_value("true") is True
        assert manager._parse_env_value("True") is True
        assert manager._parse_env_value("TRUE") is True
        assert manager._parse_env_value("yes") is True
        assert manager._parse_env_value("1") is True

        assert manager._parse_env_value("false") is False
        assert manager._parse_env_value("False") is False
        assert manager._parse_env_value("no") is False
        assert manager._parse_env_value("0") is False

    def test_parse_env_value_numeric(self) -> None:
        """Test parsing numeric environment variables."""
        manager = ConfigManager()

        assert manager._parse_env_value("42") == 42
        assert manager._parse_env_value("3.14") == 3.14
        assert manager._parse_env_value("-10") == -10
        assert manager._parse_env_value("1.5e-3") == 0.0015

    def test_parse_env_value_string(self) -> None:
        """Test parsing string environment variables."""
        manager = ConfigManager()

        assert manager._parse_env_value("hello") == "hello"
        assert manager._parse_env_value("INFO") == "INFO"
        assert manager._parse_env_value("path/to/file") == "path/to/file"

    def test_load_edge_config(self) -> None:
        """Test loading edge configuration has correct constraints."""
        config_path = Path("config/edge.yaml")
        if not config_path.exists():
            pytest.skip("Edge config file not found")

        manager = ConfigManager(config_path)
        config = manager.load()

        # Verify edge-specific settings
        assert config.edge_simulation.enabled is True
        assert config.edge_simulation.resource_constraints["cpu_cores"] == 2
        assert config.edge_simulation.resource_constraints["memory_gb"] == 2.0
        assert config.embedding.runtime.intra_op_num_threads == 2

    def test_invalid_yaml_file(self, tmp_path: Path) -> None:
        """Test handling of invalid YAML file."""
        config_file = tmp_path / "invalid.yaml"

        with open(config_file, "w") as f:
            f.write("invalid: yaml: syntax:\n  - broken")

        manager = ConfigManager(config_file)

        with pytest.raises(yaml.YAMLError):
            manager.load()

    def test_config_immutability(self, tmp_path: Path) -> None:
        """Test that loaded config is validated and cannot have invalid values."""
        config_file = tmp_path / "test_config.yaml"
        test_config = {"system": {"name": "test", "log_level": "INFO"}}

        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        manager = ConfigManager(config_file)
        config = manager.load()

        # Attempt to set invalid log level (should raise validation error on new model)
        with pytest.raises(ValidationError):
            config.system.log_level = "INVALID_LEVEL"
            config.model_validate(config.model_dump())

    def test_all_required_sections_present(self) -> None:
        """Test that default config has all required sections."""
        manager = ConfigManager()
        config = manager.load()

        # Verify all major sections exist
        assert hasattr(config, "system")
        assert hasattr(config, "ingestion")
        assert hasattr(config, "embedding")
        assert hasattr(config, "privacy")
        assert hasattr(config, "database")
        assert hasattr(config, "anomaly")
        assert hasattr(config, "dashboard")
        assert hasattr(config, "edge_simulation")
        assert hasattr(config, "telemetry")

    def test_default_privacy_settings(self) -> None:
        """Test that default privacy settings are secure."""
        manager = ConfigManager()
        config = manager.load()

        assert config.privacy.enabled is True
        assert config.privacy.differential_privacy.enabled is True
        assert config.privacy.memory_wiping.enabled is True
        assert config.privacy.budget_tracking.enabled is True
        assert config.privacy.audit_logging.enabled is True

    def test_config_file_not_found_generates_default(self, tmp_path: Path) -> None:
        """Test that missing config file triggers default generation."""
        config_file = tmp_path / "nonexistent.yaml"

        manager = ConfigManager(config_file)
        config = manager.load()

        # Should have generated default config
        assert config_file.exists()
        assert isinstance(config, KizunaConfig)
