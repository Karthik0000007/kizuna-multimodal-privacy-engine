"""Audio data simulator for Kizuna Privacy Engine.

Generates synthetic audio chunks with various scenarios:
- Normal ambient
- Shout/scream
- Glass breaking
- Silence
- Crowd murmur
"""

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from ..logger import get_ingestion_logger

logger = get_ingestion_logger()


class AudioScenario(Enum):
    """Supported audio scenarios."""

    NORMAL_AMBIENT = "normal_ambient"
    SHOUT_SCREAM = "shout_scream"
    GLASS_BREAKING = "glass_breaking"
    SILENCE = "silence"
    CROWD_MURMUR = "crowd_murmur"
    FOOTSTEPS = "footsteps"
    DOOR_SLAM = "door_slam"
    ALARM = "alarm"
    UNUSUAL_SOUND = "unusual_sound"


@dataclass
class AudioChunk:
    """Container for a single audio chunk with metadata."""

    audio: NDArray[np.float32]  # Shape: (num_samples,)
    timestamp: float
    chunk_number: int
    sample_rate: int
    scenario: AudioScenario
    metadata: dict


class AudioSimulator:
    """Synthetic audio chunk generator.

    Generates realistic audio signals for testing and development without
    requiring actual audio datasets or microphones.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        scenario: AudioScenario = AudioScenario.NORMAL_AMBIENT,
        enable_noise: bool = True,
        noise_level: float = 0.01,
    ) -> None:
        """Initialize audio simulator.

        Args:
            sample_rate: Audio sample rate in Hz (8000-48000)
            chunk_duration: Duration of each audio chunk in seconds
            scenario: Audio scenario to simulate
            enable_noise: Whether to add background noise
            noise_level: Background noise level (0.0-1.0)
        """
        if not 8000 <= sample_rate <= 48000:
            raise ValueError(f"Sample rate must be between 8000 and 48000, got {sample_rate}")
        if not 0.0 < chunk_duration <= 10.0:
            raise ValueError(f"Chunk duration must be between 0 and 10, got {chunk_duration}")
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError(f"Noise level must be between 0 and 1, got {noise_level}")

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.scenario = scenario
        self.enable_noise = enable_noise
        self.noise_level = noise_level

        self.chunk_number = 0
        self.start_time = time.time()
        self.samples_per_chunk = int(sample_rate * chunk_duration)

        # Internal state for scenario animations
        self._phase = 0.0

        logger.info(
            "audio_simulator_initialized",
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            scenario=scenario.value,
            samples_per_chunk=self.samples_per_chunk,
        )

    def generate(
        self, duration_seconds: Optional[float] = None
    ) -> Generator[AudioChunk, None, None]:
        """Generate audio chunks.

        Args:
            duration_seconds: Duration to generate (None = infinite)

        Yields:
            AudioChunk objects with synthetic audio
        """
        total_chunks = int(duration_seconds / self.chunk_duration) if duration_seconds else None

        logger.info(
            "audio_generation_started",
            duration_seconds=duration_seconds,
            total_chunks=total_chunks,
        )

        while True:
            if total_chunks and self.chunk_number >= total_chunks:
                break

            chunk_start = time.time()

            # Generate audio based on scenario
            audio = self._generate_audio()

            # Add background noise
            if self.enable_noise:
                audio = self._add_noise(audio)

            # Normalize to [-1, 1]
            audio = self._normalize(audio)

            # Create AudioChunk object
            audio_chunk = AudioChunk(
                audio=audio.astype(np.float32),
                timestamp=time.time(),
                chunk_number=self.chunk_number,
                sample_rate=self.sample_rate,
                scenario=self.scenario,
                metadata={
                    "duration": self.chunk_duration,
                    "num_samples": len(audio),
                    "rms_amplitude": float(np.sqrt(np.mean(audio**2))),
                },
            )

            self.chunk_number += 1

            yield audio_chunk

            # Maintain timing
            elapsed = time.time() - chunk_start
            sleep_time = max(0, self.chunk_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("audio_generation_completed", total_chunks=self.chunk_number)

    def _generate_audio(self) -> NDArray[np.float32]:
        """Generate a single audio chunk based on scenario."""
        if self.scenario == AudioScenario.SILENCE:
            return self._generate_silence()
        elif self.scenario == AudioScenario.NORMAL_AMBIENT:
            return self._generate_normal_ambient()
        elif self.scenario == AudioScenario.SHOUT_SCREAM:
            return self._generate_shout_scream()
        elif self.scenario == AudioScenario.GLASS_BREAKING:
            return self._generate_glass_breaking()
        elif self.scenario == AudioScenario.CROWD_MURMUR:
            return self._generate_crowd_murmur()
        elif self.scenario == AudioScenario.FOOTSTEPS:
            return self._generate_footsteps()
        elif self.scenario == AudioScenario.DOOR_SLAM:
            return self._generate_door_slam()
        elif self.scenario == AudioScenario.ALARM:
            return self._generate_alarm()
        elif self.scenario == AudioScenario.UNUSUAL_SOUND:
            return self._generate_glass_breaking()
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _generate_silence(self) -> NDArray[np.float32]:
        """Generate silence (very low amplitude noise only)."""
        return np.zeros(self.samples_per_chunk, dtype=np.float32)

    def _generate_normal_ambient(self) -> NDArray[np.float32]:
        """Generate normal ambient room sound."""
        t = np.linspace(0, self.chunk_duration, self.samples_per_chunk)

        # Low-frequency hum (HVAC, electronics)
        hum = 0.01 * np.sin(2 * np.pi * 60 * t)  # 60Hz hum

        # Pink noise (1/f noise, more natural than white noise)
        pink_noise = self._generate_pink_noise(self.samples_per_chunk) * 0.02

        # Occasional subtle variations
        variation = 0.005 * np.sin(2 * np.pi * 0.5 * t) * np.random.random()

        audio = hum + pink_noise + variation
        return audio

    def _generate_shout_scream(self) -> NDArray[np.float32]:
        """Generate shout/scream sound."""
        t = np.linspace(0, self.chunk_duration, self.samples_per_chunk)

        # Fundamental frequency (voice range: 100-300 Hz for urgency)
        fundamental = 200 + 100 * np.random.random()

        # Amplitude envelope: sudden onset, sustain, decay
        envelope = np.zeros_like(t)
        onset_samples = int(0.05 * self.sample_rate)
        sustain_samples = int(0.6 * self.sample_rate)
        decay_samples = self.samples_per_chunk - onset_samples - sustain_samples

        envelope[:onset_samples] = np.linspace(0, 1, onset_samples)
        envelope[onset_samples : onset_samples + sustain_samples] = 1.0
        envelope[onset_samples + sustain_samples :] = np.linspace(1, 0, decay_samples)

        # Voice signal with harmonics
        signal = 0.5 * np.sin(2 * np.pi * fundamental * t)  # Fundamental
        signal += 0.3 * np.sin(2 * np.pi * 2 * fundamental * t)  # 2nd harmonic
        signal += 0.2 * np.sin(2 * np.pi * 3 * fundamental * t)  # 3rd harmonic

        # Add noise for breathiness/urgency
        noise = self._generate_pink_noise(self.samples_per_chunk) * 0.2

        audio = (signal + noise) * envelope
        return audio

    def _generate_glass_breaking(self) -> NDArray[np.float32]:
        """Generate glass breaking sound."""
        t = np.linspace(0, self.chunk_duration, self.samples_per_chunk)

        # Initial impact (sharp transient)
        impact_duration = 0.1
        impact_samples = int(impact_duration * self.sample_rate)
        impact = np.zeros(self.samples_per_chunk)

        # High-frequency burst
        impact_freq = np.random.uniform(2000, 8000, impact_samples)
        impact[:impact_samples] = np.sin(2 * np.pi * np.cumsum(impact_freq) / self.sample_rate)
        impact[:impact_samples] *= np.exp(-np.linspace(0, 10, impact_samples))  # Fast decay

        # Shattering (multiple resonances decaying)
        shatter = np.zeros(self.samples_per_chunk)
        resonances = [1200, 2400, 3800, 5200, 6800]  # Glass resonant frequencies

        for freq in resonances:
            resonance = np.sin(2 * np.pi * freq * t)
            decay = np.exp(-t * 5)  # Exponential decay
            shatter += resonance * decay * np.random.uniform(0.1, 0.3)

        # Combine impact and shatter
        audio = impact * 0.8 + shatter * 0.2

        return audio

    def _generate_crowd_murmur(self) -> NDArray[np.float32]:
        """Generate crowd murmur sound."""
        t = np.linspace(0, self.chunk_duration, self.samples_per_chunk)

        # Multiple voices at different frequencies
        num_voices = 15
        murmur = np.zeros(self.samples_per_chunk)

        for _ in range(num_voices):
            # Voice fundamental frequency (80-250 Hz)
            freq = np.random.uniform(80, 250)

            # Random amplitude modulation (conversation dynamics)
            modulation_freq = np.random.uniform(0.5, 3.0)
            amplitude = 0.05 * (1 + 0.5 * np.sin(2 * np.pi * modulation_freq * t))

            # Voice signal
            voice = amplitude * np.sin(2 * np.pi * freq * t)

            # Add formants (vowel characteristics)
            formant1 = 0.03 * np.sin(2 * np.pi * freq * 3 * t)
            formant2 = 0.02 * np.sin(2 * np.pi * freq * 5 * t)

            murmur += voice + formant1 + formant2

        # Add ambient noise
        murmur += self._generate_pink_noise(self.samples_per_chunk) * 0.05

        # Slight amplitude variation (crowd dynamics)
        overall_modulation = 1 + 0.2 * np.sin(2 * np.pi * 0.3 * t)
        murmur *= overall_modulation

        return murmur

    def _generate_footsteps(self) -> NDArray[np.float32]:
        """Generate footsteps sound."""
        audio = np.zeros(self.samples_per_chunk)

        # Footstep timing (about 2 steps per second)
        step_interval = int(0.5 * self.sample_rate)
        num_steps = int(self.chunk_duration * 2)

        for i in range(num_steps):
            step_start = i * step_interval
            if step_start >= self.samples_per_chunk:
                break

            # Single footstep: impact + decay
            step_duration = int(0.15 * self.sample_rate)
            step_end = min(step_start + step_duration, self.samples_per_chunk)
            step_samples = step_end - step_start

            # Impact frequencies (low thud)
            t_step = np.linspace(0, 0.15, step_samples)
            impact = 0.3 * np.sin(2 * np.pi * 80 * t_step)
            impact += 0.2 * np.sin(2 * np.pi * 120 * t_step)

            # Exponential decay
            decay = np.exp(-t_step * 20)

            audio[step_start:step_end] += impact * decay

        return audio

    def _generate_door_slam(self) -> NDArray[np.float32]:
        """Generate door slam sound."""
        t = np.linspace(0, self.chunk_duration, self.samples_per_chunk)

        # Sharp impact at the start
        impact_duration = 0.05
        impact_samples = int(impact_duration * self.sample_rate)

        impact = np.zeros(self.samples_per_chunk)
        impact[:impact_samples] = np.exp(-np.linspace(0, 20, impact_samples))

        # Low-frequency thud
        thud_freq = 60
        thud = 0.4 * np.sin(2 * np.pi * thud_freq * t) * np.exp(-t * 10)

        # High-frequency rattle (door frame vibration)
        rattle_freq = 800
        rattle = 0.1 * np.sin(2 * np.pi * rattle_freq * t) * np.exp(-t * 15)

        audio = impact * 0.5 + thud + rattle

        return audio

    def _generate_alarm(self) -> NDArray[np.float32]:
        """Generate alarm sound."""
        t = np.linspace(0, self.chunk_duration, self.samples_per_chunk)

        # Alternating two-tone alarm (like smoke detector)
        tone1_freq = 1000  # Hz
        tone2_freq = 800  # Hz
        alternation_freq = 2  # Hz (2 times per second)

        # Square wave for alternation
        alternation = np.sign(np.sin(2 * np.pi * alternation_freq * t))

        # Generate tones
        tone1 = np.sin(2 * np.pi * tone1_freq * t)
        tone2 = np.sin(2 * np.pi * tone2_freq * t)

        # Combine based on alternation
        audio = np.where(alternation > 0, tone1, tone2) * 0.4

        # Add slight amplitude modulation for urgency
        modulation = 1 + 0.2 * np.sin(2 * np.pi * 5 * t)
        audio *= modulation

        return audio

    def _generate_pink_noise(self, num_samples: int) -> NDArray[np.float32]:
        """Generate pink noise (1/f noise).

        Args:
            num_samples: Number of samples to generate

        Returns:
            Pink noise signal
        """
        # White noise
        white = np.random.randn(num_samples)

        # Simple pink noise approximation using filtering
        # Running sum creates 1/f spectrum
        pink = np.cumsum(white)
        pink = pink - np.mean(pink)
        pink = pink / (np.std(pink) + 1e-8)

        return pink.astype(np.float32)

    def _add_noise(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Add background noise to audio.

        Args:
            audio: Input audio signal

        Returns:
            Audio with added noise
        """
        noise = np.random.randn(len(audio)).astype(np.float32) * self.noise_level
        return audio + noise

    def _normalize(
        self, audio: NDArray[np.float32], target_level: float = 0.7
    ) -> NDArray[np.float32]:
        """Normalize audio to target level.

        Args:
            audio: Input audio signal
            target_level: Target peak level (0-1)

        Returns:
            Normalized audio
        """
        peak = np.abs(audio).max()
        if peak > 1e-8:
            audio = audio / peak * target_level
        return audio

    def load_from_file(self, audio_path: Path) -> Generator[AudioChunk, None, None]:
        """Load audio chunks from an actual audio file (e.g., ESC-50 dataset).

        Args:
            audio_path: Path to audio file

        Yields:
            AudioChunk objects from the audio file

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If audio cannot be loaded
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio, original_sample_rate = sf.read(str(audio_path))
        except Exception as e:
            raise RuntimeError(f"Cannot load audio file: {audio_path}") from e

        # Resample if needed
        if original_sample_rate != self.sample_rate:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=original_sample_rate, target_sr=self.sample_rate
            )

        logger.info(
            "audio_file_loaded",
            path=str(audio_path),
            original_sample_rate=original_sample_rate,
            duration=len(audio) / self.sample_rate,
        )

        # Yield chunks
        chunk_number = 0
        for start in range(0, len(audio), self.samples_per_chunk):
            end = min(start + self.samples_per_chunk, len(audio))
            chunk = audio[start:end]

            # Pad last chunk if needed
            if len(chunk) < self.samples_per_chunk:
                chunk = np.pad(chunk, (0, self.samples_per_chunk - len(chunk)))

            audio_chunk = AudioChunk(
                audio=chunk.astype(np.float32),
                timestamp=time.time(),
                chunk_number=chunk_number,
                sample_rate=self.sample_rate,
                scenario=self.scenario,
                metadata={
                    "source": "file",
                    "path": str(audio_path),
                    "original_sample_rate": original_sample_rate,
                },
            )

            chunk_number += 1
            yield audio_chunk

            time.sleep(self.chunk_duration)

    def reset(self) -> None:
        """Reset simulator state."""
        self.chunk_number = 0
        self.start_time = time.time()
        self._phase = 0.0
        logger.info("audio_simulator_reset")


def main() -> None:
    """Demo audio simulator."""
    import argparse

    parser = argparse.ArgumentParser(description="Kizuna Audio Simulator")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--duration", type=int, default=5, help="Duration in seconds")
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal_ambient",
        choices=[s.value for s in AudioScenario],
        help="Audio scenario",
    )
    parser.add_argument("--output", type=str, help="Output audio file (optional)")
    args = parser.parse_args()

    scenario = AudioScenario(args.scenario)
    simulator = AudioSimulator(sample_rate=args.sample_rate, scenario=scenario)

    print(f"Generating {args.duration}s of {scenario.value} at {args.sample_rate} Hz...")

    all_audio = []
    for chunk_obj in simulator.generate(duration_seconds=args.duration):
        all_audio.append(chunk_obj.audio)
        print(f"Chunk {chunk_obj.chunk_number}: " f"RMS={chunk_obj.metadata['rms_amplitude']:.4f}")

    # Save to file if output specified
    if args.output:
        concatenated = np.concatenate(all_audio)
        sf.write(args.output, concatenated, args.sample_rate)
        print(f"Saved to {args.output}")

    print(f"Generated {simulator.chunk_number} chunks")


if __name__ == "__main__":
    main()
