"""Environmental sensor simulator for Kizuna Privacy Engine.

Generates synthetic environmental sensor readings:
- Temperature (°C)
- Humidity (%)
- Motion (PIR binary)
- Light level (lux)
- Air quality (AQI)
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generator, List, Optional

import numpy as np

from ..logger import get_ingestion_logger

logger = get_ingestion_logger()


class EnvironmentalScenario(Enum):
    """Supported environmental scenarios."""

    NORMAL_INDOOR = "normal_indoor"
    OCCUPIED_ROOM = "occupied_room"
    EMPTY_ROOM = "empty_room"
    HVAC_FAILURE = "hvac_failure"
    FIRE_EMERGENCY = "fire_emergency"
    WINDOW_OPEN = "window_open"
    NIGHT_TIME = "night_time"
    CROWDED_SPACE = "crowded_space"


@dataclass
class SensorReading:
    """Container for environmental sensor readings."""

    temperature: float  # Celsius
    humidity: float  # Percentage
    motion: int  # Binary (0 or 1)
    light: float  # Lux
    air_quality: float  # AQI (Air Quality Index)
    timestamp: float
    reading_number: int
    scenario: EnvironmentalScenario
    metadata: dict


class EnvironmentalSimulator:
    """Synthetic environmental sensor data generator.

    Simulates realistic sensor readings with temporal patterns:
    - Day/night cycles
    - Seasonal drift
    - Sudden spikes (anomalies)
    - Measurement noise
    """

    def __init__(
        self,
        sensors: Optional[List[str]] = None,
        polling_rate: float = 1.0,
        scenario: EnvironmentalScenario = EnvironmentalScenario.NORMAL_INDOOR,
        enable_noise: bool = True,
        noise_level: float = 0.05,
        enable_day_night_cycle: bool = True,
        anomaly_probability: float = 0.01,
    ) -> None:
        """Initialize environmental sensor simulator.

        Args:
            sensors: List of sensors to simulate (default: all)
            polling_rate: Sensor polling rate in seconds
            scenario: Environmental scenario to simulate
            enable_noise: Whether to add sensor noise
            noise_level: Sensor noise level (0.0-1.0)
            enable_day_night_cycle: Whether to simulate day/night cycles
            anomaly_probability: Probability of anomaly injection (0.0-1.0)
        """
        if sensors is None:
            sensors = ["temperature", "humidity", "motion", "light", "air_quality"]

        if not 0.0 < polling_rate <= 60.0:
            raise ValueError(f"Polling rate must be between 0 and 60, got {polling_rate}")
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError(f"Noise level must be between 0 and 1, got {noise_level}")
        if not 0.0 <= anomaly_probability <= 1.0:
            raise ValueError(f"Anomaly probability must be between 0 and 1, got {anomaly_probability}")

        self.sensors = sensors
        self.polling_rate = polling_rate
        self.scenario = scenario
        self.enable_noise = enable_noise
        self.noise_level = noise_level
        self.enable_day_night_cycle = enable_day_night_cycle
        self.anomaly_probability = anomaly_probability

        self.reading_number = 0
        self.start_time = time.time()

        # Base values for normal indoor scenario
        self._base_temperature = 22.0  # °C
        self._base_humidity = 50.0  # %
        self._base_light = 400.0  # lux
        self._base_air_quality = 50.0  # AQI

        # Internal state
        self._motion_state = 0
        self._motion_cooldown = 0

        logger.info(
            "env_simulator_initialized",
            sensors=sensors,
            polling_rate=polling_rate,
            scenario=scenario.value,
        )

    def generate(self, duration_seconds: Optional[float] = None) -> Generator[SensorReading, None, None]:
        """Generate environmental sensor readings.

        Args:
            duration_seconds: Duration to generate (None = infinite)

        Yields:
            SensorReading objects with synthetic sensor data
        """
        total_readings = int(duration_seconds / self.polling_rate) if duration_seconds else None

        logger.info(
            "env_generation_started",
            duration_seconds=duration_seconds,
            total_readings=total_readings,
        )

        while True:
            if total_readings and self.reading_number >= total_readings:
                break

            reading_start = time.time()

            # Generate sensor readings based on scenario
            values = self._generate_readings()

            # Add sensor noise
            if self.enable_noise:
                values = self._add_noise(values)

            # Inject anomalies
            if np.random.random() < self.anomaly_probability:
                values = self._inject_anomaly(values)

            # Create SensorReading object
            sensor_reading = SensorReading(
                temperature=values["temperature"],
                humidity=values["humidity"],
                motion=values["motion"],
                light=values["light"],
                air_quality=values["air_quality"],
                timestamp=time.time(),
                reading_number=self.reading_number,
                scenario=self.scenario,
                metadata={
                    "time_of_day": self._get_time_of_day(),
                    "day_phase": self._get_day_phase(),
                },
            )

            self.reading_number += 1

            yield sensor_reading

            # Maintain polling rate
            elapsed = time.time() - reading_start
            sleep_time = max(0, self.polling_rate - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("env_generation_completed", total_readings=self.reading_number)

    def _generate_readings(self) -> Dict[str, float]:
        """Generate sensor readings based on scenario."""
        if self.scenario == EnvironmentalScenario.NORMAL_INDOOR:
            return self._generate_normal_indoor()
        elif self.scenario == EnvironmentalScenario.OCCUPIED_ROOM:
            return self._generate_occupied_room()
        elif self.scenario == EnvironmentalScenario.EMPTY_ROOM:
            return self._generate_empty_room()
        elif self.scenario == EnvironmentalScenario.HVAC_FAILURE:
            return self._generate_hvac_failure()
        elif self.scenario == EnvironmentalScenario.FIRE_EMERGENCY:
            return self._generate_fire_emergency()
        elif self.scenario == EnvironmentalScenario.WINDOW_OPEN:
            return self._generate_window_open()
        elif self.scenario == EnvironmentalScenario.NIGHT_TIME:
            return self._generate_night_time()
        elif self.scenario == EnvironmentalScenario.CROWDED_SPACE:
            return self._generate_crowded_space()
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _generate_normal_indoor(self) -> Dict[str, float]:
        """Generate normal indoor environment readings."""
        # Apply day/night cycle
        temp_offset = self._get_temperature_offset()
        light_multiplier = self._get_light_multiplier()

        temperature = self._base_temperature + temp_offset
        humidity = self._base_humidity + np.sin(self.reading_number * 0.01) * 5
        motion = self._generate_motion(probability=0.3)
        light = self._base_light * light_multiplier
        air_quality = self._base_air_quality + np.sin(self.reading_number * 0.02) * 10

        return {
            "temperature": temperature,
            "humidity": humidity,
            "motion": motion,
            "light": light,
            "air_quality": air_quality,
        }

    def _generate_occupied_room(self) -> Dict[str, float]:
        """Generate occupied room readings (higher temperature, more motion)."""
        temp_offset = self._get_temperature_offset()
        light_multiplier = self._get_light_multiplier()

        # Higher temperature due to occupancy
        temperature = self._base_temperature + temp_offset + 2.0

        # Higher humidity from breath
        humidity = self._base_humidity + 10.0 + np.sin(self.reading_number * 0.02) * 3

        # Frequent motion
        motion = self._generate_motion(probability=0.8)

        # Lights on
        light = max(800.0, self._base_light * light_multiplier)

        # Slightly worse air quality
        air_quality = self._base_air_quality + 20.0 + np.sin(self.reading_number * 0.03) * 5

        return {
            "temperature": temperature,
            "humidity": humidity,
            "motion": motion,
            "light": light,
            "air_quality": air_quality,
        }

    def _generate_empty_room(self) -> Dict[str, float]:
        """Generate empty room readings (stable, no motion)."""
        temp_offset = self._get_temperature_offset()
        light_multiplier = self._get_light_multiplier()

        temperature = self._base_temperature + temp_offset - 1.0
        humidity = self._base_humidity + np.sin(self.reading_number * 0.005) * 2
        motion = 0  # No motion in empty room
        light = self._base_light * light_multiplier * 0.5  # Dimmer
        air_quality = self._base_air_quality + np.sin(self.reading_number * 0.01) * 3

        return {
            "temperature": temperature,
            "humidity": humidity,
            "motion": motion,
            "light": light,
            "air_quality": air_quality,
        }

    def _generate_hvac_failure(self) -> Dict[str, float]:
        """Generate HVAC failure readings (temperature drift)."""
        # Temperature drifts up over time
        temp_drift = (self.reading_number * 0.05)
        temperature = self._base_temperature + temp_drift

        # Humidity also drifts
        humidity = self._base_humidity + (self.reading_number * 0.03)

        motion = self._generate_motion(probability=0.4)
        light = self._base_light * self._get_light_multiplier()
        air_quality = self._base_air_quality + (self.reading_number * 0.02)

        return {
            "temperature": min(temperature, 35.0),  # Cap at 35°C
            "humidity": min(humidity, 85.0),  # Cap at 85%
            "motion": motion,
            "light": light,
            "air_quality": min(air_quality, 150.0),  # Worsening air quality
        }

    def _generate_fire_emergency(self) -> Dict[str, float]:
        """Generate fire emergency readings (rapid temperature spike)."""
        # Rapid temperature increase
        temperature = self._base_temperature + (self.reading_number * 0.5) + np.random.normal(0, 3)

        # Humidity drops (moisture evaporates)
        humidity = max(20.0, self._base_humidity - (self.reading_number * 0.2))

        # Frequent motion (people evacuating)
        motion = self._generate_motion(probability=0.9)

        # Light affected by smoke
        light = max(50.0, self._base_light - (self.reading_number * 5))

        # Severe air quality degradation
        air_quality = self._base_air_quality + (self.reading_number * 5)

        return {
            "temperature": min(temperature, 60.0),  # Cap at 60°C
            "humidity": humidity,
            "motion": motion,
            "light": light,
            "air_quality": min(air_quality, 500.0),  # Max AQI
        }

    def _generate_window_open(self) -> Dict[str, float]:
        """Generate readings with window open (external influence)."""
        temp_offset = self._get_temperature_offset()

        # Temperature influenced by outside (assume cooler)
        temperature = self._base_temperature + temp_offset - 5.0 + np.random.normal(0, 1)

        # Humidity fluctuates more
        humidity = self._base_humidity + np.random.normal(0, 10)

        motion = self._generate_motion(probability=0.3)

        # Natural light if daytime
        light_multiplier = self._get_light_multiplier()
        light = self._base_light * light_multiplier * 1.5

        # Better air quality (fresh air)
        air_quality = max(20.0, self._base_air_quality - 20.0 + np.random.normal(0, 5))

        return {
            "temperature": temperature,
            "humidity": np.clip(humidity, 30.0, 80.0),
            "motion": motion,
            "light": light,
            "air_quality": air_quality,
        }

    def _generate_night_time(self) -> Dict[str, float]:
        """Generate nighttime readings (cooler, dark, minimal motion)."""
        temperature = self._base_temperature - 3.0 + np.sin(self.reading_number * 0.01) * 1
        humidity = self._base_humidity + 5.0 + np.sin(self.reading_number * 0.02) * 3
        motion = self._generate_motion(probability=0.05)  # Very low motion
        light = 10.0 + np.random.uniform(0, 20)  # Very low light
        air_quality = self._base_air_quality + np.sin(self.reading_number * 0.01) * 5

        return {
            "temperature": temperature,
            "humidity": humidity,
            "motion": motion,
            "light": light,
            "air_quality": air_quality,
        }

    def _generate_crowded_space(self) -> Dict[str, float]:
        """Generate crowded space readings (hot, stuffy, constant motion)."""
        temp_offset = self._get_temperature_offset()

        # Much higher temperature
        temperature = self._base_temperature + temp_offset + 5.0 + np.random.uniform(0, 2)

        # High humidity
        humidity = min(80.0, self._base_humidity + 20.0 + np.random.uniform(0, 5))

        # Constant motion
        motion = 1

        # Bright lights
        light = 800.0 + np.random.uniform(0, 200)

        # Poor air quality
        air_quality = self._base_air_quality + 50.0 + np.random.uniform(0, 20)

        return {
            "temperature": temperature,
            "humidity": humidity,
            "motion": motion,
            "light": light,
            "air_quality": min(air_quality, 200.0),
        }

    def _generate_motion(self, probability: float) -> int:
        """Generate motion sensor reading (binary).

        Args:
            probability: Probability of motion detection (0-1)

        Returns:
            1 if motion detected, 0 otherwise
        """
        # Cooldown after motion detected (PIR sensors have ~2s cooldown)
        if self._motion_cooldown > 0:
            self._motion_cooldown -= 1
            return self._motion_state

        # Check for new motion
        if np.random.random() < probability:
            self._motion_state = 1
            self._motion_cooldown = int(2.0 / self.polling_rate)  # 2 second cooldown
        else:
            self._motion_state = 0

        return self._motion_state

    def _get_temperature_offset(self) -> float:
        """Get temperature offset based on day/night cycle."""
        if not self.enable_day_night_cycle:
            return 0.0

        # Simulate 24-hour cycle (compressed into simulation time)
        cycle_period = 300  # readings for a full cycle
        phase = (self.reading_number % cycle_period) / cycle_period * 2 * np.pi

        # Temperature varies ±3°C through the day
        return 3.0 * np.sin(phase - np.pi / 2)  # Coldest at "night", warmest at "day"

    def _get_light_multiplier(self) -> float:
        """Get light level multiplier based on day/night cycle."""
        if not self.enable_day_night_cycle:
            return 1.0

        cycle_period = 300
        phase = (self.reading_number % cycle_period) / cycle_period * 2 * np.pi

        # Light varies from 0.1x (night) to 2x (day)
        return max(0.1, 1.0 + 0.9 * np.sin(phase))

    def _get_time_of_day(self) -> str:
        """Get time of day label."""
        if not self.enable_day_night_cycle:
            return "constant"

        cycle_period = 300
        phase = (self.reading_number % cycle_period) / cycle_period

        if phase < 0.25:
            return "night"
        elif phase < 0.5:
            return "morning"
        elif phase < 0.75:
            return "afternoon"
        else:
            return "evening"

    def _get_day_phase(self) -> float:
        """Get day phase as value between 0 and 1."""
        cycle_period = 300
        return (self.reading_number % cycle_period) / cycle_period

    def _add_noise(self, values: Dict[str, float]) -> Dict[str, float]:
        """Add realistic sensor noise.

        Args:
            values: Sensor readings dictionary

        Returns:
            Readings with added noise
        """
        noisy_values = values.copy()

        # Temperature: ±0.5°C sensor accuracy
        noisy_values["temperature"] += np.random.normal(0, 0.5 * self.noise_level)

        # Humidity: ±3% sensor accuracy
        noisy_values["humidity"] += np.random.normal(0, 3.0 * self.noise_level)

        # Motion: binary, no noise

        # Light: ±5% sensor accuracy
        noisy_values["light"] += np.random.normal(0, noisy_values["light"] * 0.05 * self.noise_level)
        noisy_values["light"] = max(0, noisy_values["light"])

        # Air quality: ±5 AQI
        noisy_values["air_quality"] += np.random.normal(0, 5.0 * self.noise_level)

        # Clip to valid ranges
        noisy_values["temperature"] = np.clip(noisy_values["temperature"], 15.0, 40.0)
        noisy_values["humidity"] = np.clip(noisy_values["humidity"], 20.0, 90.0)
        noisy_values["air_quality"] = np.clip(noisy_values["air_quality"], 0.0, 500.0)

        return noisy_values

    def _inject_anomaly(self, values: Dict[str, float]) -> Dict[str, float]:
        """Inject sudden anomaly into sensor readings.

        Args:
            values: Normal sensor readings

        Returns:
            Readings with injected anomaly
        """
        anomalous_values = values.copy()

        # Random spike in one sensor
        sensor_to_spike = np.random.choice(["temperature", "humidity", "air_quality"])

        if sensor_to_spike == "temperature":
            anomalous_values["temperature"] += np.random.uniform(5, 10)
        elif sensor_to_spike == "humidity":
            anomalous_values["humidity"] += np.random.uniform(10, 20)
        elif sensor_to_spike == "air_quality":
            anomalous_values["air_quality"] += np.random.uniform(50, 100)

        logger.warning(
            "anomaly_injected",
            sensor=sensor_to_spike,
            reading_number=self.reading_number,
        )

        return anomalous_values

    def reset(self) -> None:
        """Reset simulator state."""
        self.reading_number = 0
        self.start_time = time.time()
        self._motion_state = 0
        self._motion_cooldown = 0
        logger.info("env_simulator_reset")


def main() -> None:
    """Demo environmental sensor simulator."""
    import argparse

    parser = argparse.ArgumentParser(description="Kizuna Environmental Sensor Simulator")
    parser.add_argument("--polling-rate", type=float, default=1.0, help="Polling rate in seconds")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal_indoor",
        choices=[s.value for s in EnvironmentalScenario],
        help="Environmental scenario",
    )
    args = parser.parse_args()

    scenario = EnvironmentalScenario(args.scenario)
    simulator = EnvironmentalSimulator(polling_rate=args.polling_rate, scenario=scenario)

    print(f"Generating {args.duration}s of {scenario.value} at {args.polling_rate}s polling rate...")
    print(f"{'Time':<10} {'Temp(°C)':<10} {'Humid(%)':<10} {'Motion':<8} {'Light(lux)':<12} {'AQI':<8}")
    print("-" * 70)

    for reading in simulator.generate(duration_seconds=args.duration):
        print(
            f"{reading.timestamp - simulator.start_time:>8.1f}s "
            f"{reading.temperature:>8.1f}   "
            f"{reading.humidity:>8.1f}   "
            f"{reading.motion:>6d}   "
            f"{reading.light:>10.1f}   "
            f"{reading.air_quality:>6.1f}"
        )

    print(f"\nGenerated {simulator.reading_number} readings")


if __name__ == "__main__":
    main()
