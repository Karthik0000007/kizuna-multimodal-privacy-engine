"""Video data simulator for Kizuna Privacy Engine.

Generates synthetic video frames with various scenarios:
- Person walking
- Person falling
- Crowd movement
- Empty room
"""

import time
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ..logger import get_ingestion_logger

logger = get_ingestion_logger()


class VideoScenario(Enum):
    """Supported video scenarios."""

    PERSON_WALKING = "person_walking"
    PERSON_FALLING = "person_falling"
    CROWD_MOVEMENT = "crowd_movement"
    EMPTY_ROOM = "empty_room"
    SITTING_STILL = "sitting_still"
    WANDERING = "wandering"


@dataclass
class VideoFrame:
    """Container for a single video frame with metadata."""

    frame: NDArray[np.uint8]  # Shape: (H, W, 3)
    timestamp: float
    frame_number: int
    scenario: VideoScenario
    metadata: dict


class VideoSimulator:
    """Synthetic video frame generator.

    Generates realistic video frames for testing and development without
    requiring actual video datasets or cameras.
    """

    def __init__(
        self,
        fps: int = 15,
        resolution: tuple[int, int] = (320, 320),
        scenario: VideoScenario = VideoScenario.PERSON_WALKING,
        enable_noise: bool = True,
        noise_level: float = 0.02,
    ) -> None:
        """Initialize video simulator.

        Args:
            fps: Frames per second (1-60)
            resolution: Frame resolution as (width, height)
            scenario: Video scenario to simulate
            enable_noise: Whether to add realistic noise to frames
            noise_level: Gaussian noise level (0.0-1.0)
        """
        if not 1 <= fps <= 60:
            raise ValueError(f"FPS must be between 1 and 60, got {fps}")
        if len(resolution) != 2 or any(r <= 0 for r in resolution):
            raise ValueError(f"Invalid resolution: {resolution}")
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError(f"Noise level must be between 0 and 1, got {noise_level}")

        self.fps = fps
        self.resolution = resolution
        self.scenario = scenario
        self.enable_noise = enable_noise
        self.noise_level = noise_level

        self.frame_number = 0
        self.start_time = time.time()

        # Internal state for scenario animations
        self._person_x = resolution[0] // 4
        self._person_y = resolution[1] // 2
        self._person_velocity_x = 2
        self._person_velocity_y = 0
        self._fall_progress = 0.0
        self._wander_angle = 0.0

        logger.info(
            "video_simulator_initialized",
            fps=fps,
            resolution=resolution,
            scenario=scenario.value,
        )

    def generate(self, duration_seconds: float | None = None) -> Generator[VideoFrame, None, None]:
        """Generate video frames.

        Args:
            duration_seconds: Duration to generate (None = infinite)

        Yields:
            VideoFrame objects with synthetic frames
        """
        total_frames = int(duration_seconds * self.fps) if duration_seconds else None
        frame_time = 1.0 / self.fps

        logger.info(
            "video_generation_started",
            duration_seconds=duration_seconds,
            total_frames=total_frames,
        )

        while True:
            if total_frames and self.frame_number >= total_frames:
                break

            frame_start = time.time()

            # Generate frame based on scenario
            frame = self._generate_frame()

            # Add realistic noise
            if self.enable_noise:
                frame = self._add_noise(frame)

            # Create VideoFrame object
            video_frame = VideoFrame(
                frame=frame,
                timestamp=time.time(),
                frame_number=self.frame_number,
                scenario=self.scenario,
                metadata={
                    "fps": self.fps,
                    "resolution": self.resolution,
                    "person_position": (self._person_x, self._person_y),
                },
            )

            self.frame_number += 1

            yield video_frame

            # Maintain FPS timing
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("video_generation_completed", total_frames=self.frame_number)

    def _generate_frame(self) -> NDArray[np.uint8]:
        """Generate a single frame based on scenario."""
        if self.scenario == VideoScenario.EMPTY_ROOM:
            return self._generate_empty_room()
        elif self.scenario == VideoScenario.PERSON_WALKING:
            return self._generate_person_walking()
        elif self.scenario == VideoScenario.PERSON_FALLING:
            return self._generate_person_falling()
        elif self.scenario == VideoScenario.CROWD_MOVEMENT:
            return self._generate_crowd_movement()
        elif self.scenario == VideoScenario.SITTING_STILL:
            return self._generate_sitting_still()
        elif self.scenario == VideoScenario.WANDERING:
            return self._generate_wandering()
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _generate_empty_room(self) -> NDArray[np.uint8]:
        """Generate empty room frame."""
        width, height = self.resolution
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray

        # Add floor gradient
        for y in range(height):
            intensity = int(200 + (y / height) * 55)
            frame[y, :] = [intensity, intensity - 10, intensity - 20]

        # Add simple furniture (rectangle as table)
        cv2.rectangle(
            frame,
            (width // 3, height // 2),
            (2 * width // 3, 2 * height // 3),
            (139, 69, 19),  # Brown
            -1,
        )

        return frame

    def _generate_person_walking(self) -> NDArray[np.uint8]:
        """Generate person walking frame."""
        frame = self._generate_empty_room()

        # Update person position (walking left to right)
        self._person_x += self._person_velocity_x
        if self._person_x > self.resolution[0] - 30:
            self._person_x = 30
            self._person_y = int(self.resolution[1] * (0.3 + 0.4 * np.random.random()))

        # Draw person as ellipse (representing head/body)
        person_color = (50, 100, 200)  # Blue shirt
        cv2.ellipse(
            frame,
            (self._person_x, self._person_y),
            (15, 40),  # Width, height
            0,  # Rotation
            0,
            360,
            person_color,
            -1,
        )

        # Draw head
        cv2.circle(frame, (self._person_x, self._person_y - 50), 12, (200, 180, 160), -1)

        # Add simple leg animation (oscillating lines)
        leg_offset = int(10 * np.sin(self.frame_number * 0.3))
        cv2.line(
            frame,
            (self._person_x, self._person_y + 40),
            (self._person_x - 10 + leg_offset, self._person_y + 70),
            person_color,
            3,
        )
        cv2.line(
            frame,
            (self._person_x, self._person_y + 40),
            (self._person_x + 10 - leg_offset, self._person_y + 70),
            person_color,
            3,
        )

        return frame

    def _generate_person_falling(self) -> NDArray[np.uint8]:
        """Generate person falling frame (fall detection scenario)."""
        frame = self._generate_empty_room()

        # Fall animation: standing -> leaning -> on ground
        self._fall_progress = min(1.0, self._fall_progress + 0.05)

        person_color = (50, 100, 200)

        if self._fall_progress < 0.3:
            # Standing
            cv2.ellipse(
                frame, (self._person_x, self._person_y), (15, 40), 0, 0, 360, person_color, -1
            )
            cv2.circle(frame, (self._person_x, self._person_y - 50), 12, (200, 180, 160), -1)
        elif self._fall_progress < 0.6:
            # Leaning / falling
            lean_angle = int((self._fall_progress - 0.3) * 300)
            cv2.ellipse(
                frame,
                (self._person_x, self._person_y),
                (15, 40),
                lean_angle,
                0,
                360,
                person_color,
                -1,
            )
            head_offset = int(30 * np.sin(np.radians(lean_angle)))
            cv2.circle(
                frame,
                (self._person_x + head_offset, self._person_y - 40),
                12,
                (200, 180, 160),
                -1,
            )
        else:
            # On ground
            cv2.ellipse(
                frame,
                (self._person_x, self._person_y + 20),
                (40, 15),
                0,
                0,
                360,
                person_color,
                -1,
            )
            cv2.circle(frame, (self._person_x - 45, self._person_y + 20), 12, (200, 180, 160), -1)

        return frame

    def _generate_crowd_movement(self) -> NDArray[np.uint8]:
        """Generate crowd movement frame."""
        frame = self._generate_empty_room()

        # Draw multiple people at different positions
        num_people = 8
        for i in range(num_people):
            x = int(
                30
                + (self.resolution[0] - 60)
                * ((i + self.frame_number * 0.01) % num_people)
                / num_people
            )
            y = int(self.resolution[1] * (0.3 + 0.3 * np.sin(i * 0.7)))

            # Vary person colors
            color_variation = int(50 * np.sin(i))
            person_color = (50 + color_variation, 100, 200 - color_variation)

            # Draw simplified person
            cv2.ellipse(frame, (x, y), (12, 30), 0, 0, 360, person_color, -1)
            cv2.circle(frame, (x, y - 35), 10, (200, 180, 160), -1)

        return frame

    def _generate_sitting_still(self) -> NDArray[np.uint8]:
        """Generate sitting still frame."""
        frame = self._generate_empty_room()

        # Person sitting on chair (represented as rectangle + person)
        chair_x = self.resolution[0] // 2
        chair_y = self.resolution[1] // 2

        # Chair
        cv2.rectangle(
            frame,
            (chair_x - 25, chair_y),
            (chair_x + 25, chair_y + 50),
            (101, 67, 33),  # Dark brown
            -1,
        )

        # Person sitting
        person_color = (50, 100, 200)
        cv2.ellipse(frame, (chair_x, chair_y - 10), (15, 30), 0, 0, 360, person_color, -1)
        cv2.circle(frame, (chair_x, chair_y - 45), 12, (200, 180, 160), -1)

        # Very slight movement (breathing)
        movement = int(2 * np.sin(self.frame_number * 0.1))
        frame = np.roll(frame, movement, axis=0)

        return frame

    def _generate_wandering(self) -> NDArray[np.uint8]:
        """Generate wandering frame (elderly wandering scenario)."""
        frame = self._generate_empty_room()

        # Wandering pattern: random walk with smooth direction changes
        self._wander_angle += np.random.normal(0, 0.2)
        self._person_x += int(3 * np.cos(self._wander_angle))
        self._person_y += int(3 * np.sin(self._wander_angle))

        # Boundary constraints
        self._person_x = np.clip(self._person_x, 30, self.resolution[0] - 30)
        self._person_y = np.clip(self._person_y, 80, self.resolution[1] - 30)

        # Draw person with uncertain movement trail
        person_color = (150, 100, 100)  # Muted color for elderly

        # Trail effect (previous positions)
        for i in range(5):
            alpha = (5 - i) / 5
            trail_x = int(self._person_x - i * 5 * np.cos(self._wander_angle))
            trail_y = int(self._person_y - i * 5 * np.sin(self._wander_angle))
            color_faded = tuple(int(c * alpha) for c in person_color)
            cv2.circle(frame, (trail_x, trail_y), 10, color_faded, -1)

        # Main person
        cv2.ellipse(frame, (self._person_x, self._person_y), (12, 35), 0, 0, 360, person_color, -1)
        cv2.circle(frame, (self._person_x, self._person_y - 40), 11, (200, 180, 160), -1)

        return frame

    def _add_noise(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Add realistic Gaussian noise to frame.

        Args:
            frame: Input frame

        Returns:
            Frame with added noise
        """
        noise = np.random.normal(0, self.noise_level * 255, frame.shape).astype(np.int16)
        noisy_frame = frame.astype(np.int16) + noise
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
        return noisy_frame

    def load_from_file(self, video_path: Path) -> Generator[VideoFrame, None, None]:
        """Load frames from an actual video file (e.g., UP-Fall dataset).

        Args:
            video_path: Path to video file

        Yields:
            VideoFrame objects from the video file

        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If video cannot be opened
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info("video_file_loaded", path=str(video_path), fps=actual_fps)

        frame_number = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize to target resolution
                frame = cv2.resize(frame, self.resolution)

                video_frame = VideoFrame(
                    frame=frame,
                    timestamp=time.time(),
                    frame_number=frame_number,
                    scenario=self.scenario,
                    metadata={
                        "source": "file",
                        "path": str(video_path),
                        "original_fps": actual_fps,
                    },
                )

                frame_number += 1
                yield video_frame

                # Maintain target FPS
                time.sleep(1.0 / self.fps)

        finally:
            cap.release()
            logger.info("video_file_released", frames_read=frame_number)

    def reset(self) -> None:
        """Reset simulator state."""
        self.frame_number = 0
        self.start_time = time.time()
        self._person_x = self.resolution[0] // 4
        self._person_y = self.resolution[1] // 2
        self._fall_progress = 0.0
        self._wander_angle = 0.0
        logger.info("video_simulator_reset")


def main() -> None:
    """Demo video simulator."""
    import argparse

    parser = argparse.ArgumentParser(description="Kizuna Video Simulator")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument(
        "--scenario",
        type=str,
        default="person_walking",
        choices=[s.value for s in VideoScenario],
        help="Video scenario",
    )
    parser.add_argument("--output", type=str, help="Output video file (optional)")
    args = parser.parse_args()

    scenario = VideoScenario(args.scenario)
    simulator = VideoSimulator(fps=args.fps, scenario=scenario)

    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, simulator.resolution)

    print(f"Generating {args.duration}s of {scenario.value} at {args.fps} FPS...")

    for frame_obj in simulator.generate(duration_seconds=args.duration):
        # Display frame
        cv2.imshow("Kizuna Video Simulator", frame_obj.frame)

        # Write to file if output specified
        if video_writer:
            video_writer.write(frame_obj.frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    print(f"Generated {simulator.frame_number} frames")


if __name__ == "__main__":
    main()
