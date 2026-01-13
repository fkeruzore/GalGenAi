"""Tests for beta scheduling utilities."""

import pytest

from galgenai.training import Schedule, ScheduleConfig, ScheduleType


class TestScheduleConfig:
    """Tests for ScheduleConfig dataclass."""

    def test_default_values(self):
        config = ScheduleConfig()
        assert config.schedule_type == "constant"
        assert config.initial_value == 0.0
        assert config.final_value == 1.0
        assert config.warmup_steps == 0
        assert config.total_steps is None
        assert config.num_cycles == 4

    def test_custom_values(self):
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=0.1,
            final_value=0.9,
            total_steps=1000,
        )
        assert config.schedule_type == "linear"
        assert config.initial_value == 0.1
        assert config.final_value == 0.9
        assert config.total_steps == 1000


class TestScheduleType:
    """Tests for ScheduleType enum."""

    def test_enum_values(self):
        assert ScheduleType.CONSTANT.value == "constant"
        assert ScheduleType.LINEAR.value == "linear"
        assert ScheduleType.COSINE.value == "cosine"
        assert ScheduleType.WARMUP_COSINE.value == "warmup_cosine"
        assert ScheduleType.CYCLICAL.value == "cyclical"


class TestConstantSchedule:
    """Tests for constant schedule."""

    def test_float_input(self):
        schedule = Schedule(1.0)
        assert schedule.get_value(0) == 1.0
        assert schedule.get_value(100) == 1.0
        assert schedule.get_value(10000) == 1.0

    def test_int_input(self):
        schedule = Schedule(5)
        assert schedule.get_value(0) == 5.0

    def test_config_constant(self):
        config = ScheduleConfig(
            schedule_type="constant",
            final_value=0.5,
        )
        schedule = Schedule(config)
        assert schedule.get_value(0) == 0.5
        assert schedule.get_value(1000) == 0.5


class TestLinearSchedule:
    """Tests for linear schedule."""

    def test_linear_basic(self):
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=0.0,
            final_value=1.0,
            total_steps=100,
        )
        schedule = Schedule(config)
        assert schedule.get_value(0) == 0.0
        assert abs(schedule.get_value(50) - 0.5) < 0.001
        assert schedule.get_value(100) == 1.0

    def test_linear_reverse(self):
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=1.0,
            final_value=0.0,
            total_steps=100,
        )
        schedule = Schedule(config)
        assert schedule.get_value(0) == 1.0
        assert abs(schedule.get_value(50) - 0.5) < 0.001
        assert schedule.get_value(100) == 0.0

    def test_linear_clamps_at_end(self):
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=0.0,
            final_value=1.0,
            total_steps=100,
        )
        schedule = Schedule(config)
        # Beyond total_steps should clamp to final
        assert schedule.get_value(200) == 1.0

    def test_linear_no_total_steps(self):
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=0.0,
            final_value=1.0,
            total_steps=None,
        )
        schedule = Schedule(config)
        # Should return final_value immediately
        assert schedule.get_value(0) == 1.0


class TestCosineSchedule:
    """Tests for cosine schedule."""

    def test_cosine_basic(self):
        config = ScheduleConfig(
            schedule_type="cosine",
            initial_value=0.0,
            final_value=1.0,
            total_steps=100,
        )
        schedule = Schedule(config)
        assert schedule.get_value(0) == 0.0
        assert schedule.get_value(100) == 1.0
        # Cosine should be slower at start (convex curve)
        assert schedule.get_value(25) < 0.25

    def test_cosine_midpoint(self):
        config = ScheduleConfig(
            schedule_type="cosine",
            initial_value=0.0,
            final_value=1.0,
            total_steps=100,
        )
        schedule = Schedule(config)
        # At midpoint, cosine should be exactly 0.5
        assert abs(schedule.get_value(50) - 0.5) < 0.001


class TestWarmupCosineSchedule:
    """Tests for warmup + cosine schedule."""

    def test_warmup_phase(self):
        config = ScheduleConfig(
            schedule_type="warmup_cosine",
            initial_value=0.0,
            final_value=1.0,
            warmup_steps=100,
            total_steps=200,
        )
        schedule = Schedule(config)
        assert schedule.get_value(0) == 0.0
        assert abs(schedule.get_value(50) - 0.5) < 0.001
        assert schedule.get_value(100) == 1.0

    def test_after_warmup(self):
        config = ScheduleConfig(
            schedule_type="warmup_cosine",
            initial_value=0.0,
            final_value=1.0,
            warmup_steps=100,
            total_steps=200,
        )
        schedule = Schedule(config)
        # After warmup, should hold at final_value
        assert schedule.get_value(150) == 1.0
        assert schedule.get_value(200) == 1.0


class TestCyclicalSchedule:
    """Tests for cyclical annealing schedule."""

    def test_cyclical_basic(self):
        config = ScheduleConfig(
            schedule_type="cyclical",
            initial_value=0.0,
            final_value=1.0,
            total_steps=100,
            num_cycles=4,
        )
        schedule = Schedule(config)
        # Each cycle is 25 steps
        assert schedule.get_value(0) == 0.0
        # Start of each new cycle should reset to initial
        assert abs(schedule.get_value(25) - 0.0) < 0.001
        assert abs(schedule.get_value(50) - 0.0) < 0.001
        assert abs(schedule.get_value(75) - 0.0) < 0.001

    def test_cyclical_midcycle(self):
        config = ScheduleConfig(
            schedule_type="cyclical",
            initial_value=0.0,
            final_value=1.0,
            total_steps=100,
            num_cycles=4,
        )
        schedule = Schedule(config)
        # Midpoint of first cycle (step 12-13 of 25)
        assert 0.4 < schedule.get_value(12) < 0.6


class TestCustomSchedule:
    """Tests for custom callable schedules."""

    def test_lambda_schedule(self):
        schedule = Schedule(lambda step: min(step / 100, 1.0))
        assert schedule.get_value(0) == 0.0
        assert schedule.get_value(50) == 0.5
        assert schedule.get_value(100) == 1.0
        assert schedule.get_value(200) == 1.0

    def test_function_schedule(self):
        def custom_fn(step):
            return 0.5 if step < 50 else 1.0

        schedule = Schedule(custom_fn)
        assert schedule.get_value(0) == 0.5
        assert schedule.get_value(49) == 0.5
        assert schedule.get_value(50) == 1.0


class TestScheduleStepMethod:
    """Tests for Schedule.step() method."""

    def test_step_advances(self):
        schedule = Schedule(0.5)
        assert schedule._step == 0
        val = schedule.step()
        assert val == 0.5
        assert schedule._step == 1

    def test_step_with_linear(self):
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=0.0,
            final_value=1.0,
            total_steps=10,
        )
        schedule = Schedule(config)

        values = [schedule.step() for _ in range(11)]
        assert values[0] == 0.0
        assert abs(values[5] - 0.5) < 0.001
        assert values[10] == 1.0

    def test_set_step(self):
        schedule = Schedule(1.0)
        schedule.set_step(50)
        assert schedule._step == 50


class TestScheduleStatePersistence:
    """Tests for checkpoint save/restore."""

    def test_state_dict(self):
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=0.0,
            final_value=1.0,
            total_steps=100,
        )
        schedule = Schedule(config)
        schedule.step()
        schedule.step()

        state = schedule.state_dict()
        assert state["step"] == 2

    def test_load_state_dict(self):
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=0.0,
            final_value=1.0,
            total_steps=100,
        )
        schedule1 = Schedule(config)
        schedule1.step()
        schedule1.step()
        schedule1.step()
        state = schedule1.state_dict()

        schedule2 = Schedule(config)
        assert schedule2._step == 0
        schedule2.load_state_dict(state)
        assert schedule2._step == 3

    def test_resume_continues_correctly(self):
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=0.0,
            final_value=1.0,
            total_steps=100,
        )
        schedule1 = Schedule(config)
        for _ in range(50):
            schedule1.step()
        state = schedule1.state_dict()
        val1 = schedule1.get_value()

        schedule2 = Schedule(config)
        schedule2.load_state_dict(state)
        val2 = schedule2.get_value()

        assert abs(val1 - val2) < 0.001
        assert abs(val2 - 0.5) < 0.001


class TestScheduleEdgeCases:
    """Tests for edge cases."""

    def test_zero_total_steps(self):
        config = ScheduleConfig(
            schedule_type="linear",
            initial_value=0.0,
            final_value=1.0,
            total_steps=0,
        )
        schedule = Schedule(config)
        # Should return final_value
        assert schedule.get_value(0) == 1.0

    def test_unknown_schedule_type(self):
        config = ScheduleConfig(schedule_type="unknown")
        with pytest.raises(ValueError, match="Unknown schedule type"):
            Schedule(config)

    def test_get_value_uses_current_step_if_none(self):
        schedule = Schedule(1.0)
        schedule._step = 5
        # get_value() without arg should use current step
        val = schedule.get_value()
        assert val == 1.0
