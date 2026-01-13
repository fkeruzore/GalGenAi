"""Scheduling utilities for beta annealing in VAE training."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union


class ScheduleType(Enum):
    """Built-in schedule types."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    WARMUP_COSINE = "warmup_cosine"
    CYCLICAL = "cyclical"


@dataclass
class ScheduleConfig:
    """Configuration for a parameter schedule.

    Attributes:
        schedule_type: Type of schedule ("constant", "linear", "cosine",
            "warmup_cosine", "cyclical")
        initial_value: Starting value (for warmup schedules)
        final_value: Target value (for warmup/decay schedules)
        warmup_steps: Number of warmup steps (for warmup schedules)
        total_steps: Total training steps (auto-computed if None)
        num_cycles: Number of cycles (for cyclical schedules)
    """

    schedule_type: str = "constant"
    initial_value: float = 0.0
    final_value: float = 1.0
    warmup_steps: int = 0
    total_steps: Optional[int] = None
    num_cycles: int = 4


class Schedule:
    """Parameter scheduler that returns value based on current step.

    Can be used for beta scheduling or any other parameter that changes
    during training. Supports built-in schedules and custom functions.

    Example:
        # Using built-in schedule
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=0.0,
            final_value=1.0,
            total_steps=10000
        )
        schedule = Schedule(config)

        # Using custom function
        schedule = Schedule(lambda step: min(step / 1000, 1.0))

        # Using constant value
        schedule = Schedule(1.0)

        # In training loop
        for step in range(total_steps):
            beta = schedule.get_value(step)
            # or use schedule.step() pattern
    """

    def __init__(
        self,
        config_or_fn: Union[ScheduleConfig, Callable[[int], float], float],
    ):
        """Initialize schedule from config, callable, or constant value.

        Args:
            config_or_fn: ScheduleConfig, callable(step) -> value,
                or float for constant schedule.
        """
        self._step = 0
        self._last_value: Optional[float] = None

        if isinstance(config_or_fn, (int, float)):
            # Constant schedule
            self._fn: Callable[[int], float] = lambda step: float(config_or_fn)
            self._config: Optional[ScheduleConfig] = None
        elif callable(config_or_fn):
            # Custom function
            self._fn = config_or_fn
            self._config = None
        else:
            # ScheduleConfig
            self._config = config_or_fn
            self._fn = self._build_schedule_fn(config_or_fn)

    def _build_schedule_fn(
        self, config: ScheduleConfig
    ) -> Callable[[int], float]:
        """Build schedule function from config."""
        stype = config.schedule_type.lower()

        if stype == "constant":
            return lambda step: config.final_value

        elif stype == "linear":

            def linear_fn(step: int) -> float:
                if config.total_steps is None or config.total_steps == 0:
                    return config.final_value
                progress = min(step / config.total_steps, 1.0)
                return config.initial_value + progress * (
                    config.final_value - config.initial_value
                )

            return linear_fn

        elif stype == "cosine":

            def cosine_fn(step: int) -> float:
                if config.total_steps is None or config.total_steps == 0:
                    return config.final_value
                progress = min(step / config.total_steps, 1.0)
                # Cosine annealing from initial to final
                cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
                return config.initial_value + cosine_factor * (
                    config.final_value - config.initial_value
                )

            return cosine_fn

        elif stype == "warmup_cosine":

            def warmup_cosine_fn(step: int) -> float:
                if step < config.warmup_steps:
                    # Linear warmup
                    progress = step / max(config.warmup_steps, 1)
                    return config.initial_value + progress * (
                        config.final_value - config.initial_value
                    )
                else:
                    # Hold at final_value after warmup
                    return config.final_value

            return warmup_cosine_fn

        elif stype == "cyclical":
            # Cyclical annealing (Fu et al., 2019)
            def cyclical_fn(step: int) -> float:
                if config.total_steps is None or config.num_cycles <= 0:
                    return config.final_value
                cycle_length = config.total_steps // config.num_cycles
                if cycle_length == 0:
                    return config.final_value
                # Position within current cycle
                cycle_pos = step % cycle_length
                progress = cycle_pos / cycle_length
                # Linear warmup within each cycle
                return config.initial_value + progress * (
                    config.final_value - config.initial_value
                )

            return cyclical_fn

        else:
            raise ValueError(f"Unknown schedule type: {stype}")

    def get_value(self, step: Optional[int] = None) -> float:
        """Get schedule value at given step (or current step)."""
        if step is None:
            step = self._step
        self._last_value = self._fn(step)
        return self._last_value

    def step(self) -> float:
        """Advance schedule by one step and return new value."""
        value = self.get_value(self._step)
        self._step += 1
        return value

    def set_step(self, step: int) -> None:
        """Set current step (for resuming from checkpoint)."""
        self._step = step

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {"step": self._step}

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self._step = state.get("step", 0)
