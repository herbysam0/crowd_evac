"""Tests for crowd_evac.application.rng."""
from __future__ import annotations

import numpy as np
import pytest

from crowd_evac.application.rng import SeededRNG


class TestSeededRNG:
    """Test suite for SeededRNG reproducibility and interface."""

    # -- Happy path: reproducibility ----------------------------------------

    def test_same_seed_reproduces_sequence(self) -> None:
        """Verify same seed produces identical draw sequences."""
        seed = 42
        rng1 = SeededRNG(seed=seed)
        rng2 = SeededRNG(seed=seed)

        # Draw from both and verify they match
        seq1 = rng1.draw_floats(10)
        seq2 = rng2.draw_floats(10)
        np.testing.assert_array_equal(seq1, seq2)

    def test_same_seed_reproduces_mixed_draws(self) -> None:
        """Verify reproducibility across mixed draw types."""
        seed = 123
        rng1 = SeededRNG(seed=seed)
        rng2 = SeededRNG(seed=seed)

        # Mix of different draw operations
        f1 = rng1.draw_floats((3, 2))
        n1 = rng1.draw_randn((2,))
        u1 = rng1.draw_uniform(0, 10, 5)

        f2 = rng2.draw_floats((3, 2))
        n2 = rng2.draw_randn((2,))
        u2 = rng2.draw_uniform(0, 10, 5)

        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(n1, n2)
        np.testing.assert_array_equal(u1, u2)

    def test_different_seeds_produce_different_sequences(self) -> None:
        """Verify different seeds produce different draws."""
        rng1 = SeededRNG(seed=1)
        rng2 = SeededRNG(seed=2)

        seq1 = rng1.draw_floats(10)
        seq2 = rng2.draw_floats(10)

        # Sequences should differ (probability of collision negligible)
        assert not np.allclose(seq1, seq2)

    def test_none_seed_is_allowed(self) -> None:
        """Verify seed=None produces a valid RNG."""
        rng = SeededRNG(seed=None)
        assert rng.seed is None
        seq = rng.draw_floats(5)
        assert seq.shape == (5,)
        assert np.all((seq >= 0) & (seq < 1))

    # -- Edge cases: bounds and shapes ----------------------------------------

    def test_draw_floats_returns_correct_shape(self) -> None:
        """Verify draw_floats returns requested shape."""
        rng = SeededRNG(seed=42)

        # 1D
        arr = rng.draw_floats(10)
        assert arr.shape == (10,)

        # 2D
        arr = rng.draw_floats((3, 4))
        assert arr.shape == (3, 4)

    def test_draw_floats_in_range(self) -> None:
        """Verify draw_floats values are in [0, 1)."""
        rng = SeededRNG(seed=42)
        arr = rng.draw_floats((100,))
        assert np.all(arr >= 0)
        assert np.all(arr < 1)

    def test_draw_randn_is_normal_distribution(self) -> None:
        """Verify draw_randn produces standard normal values."""
        rng = SeededRNG(seed=42)
        arr = rng.draw_randn((1000,))

        # Mean close to 0, std close to 1 (lenient due to finite sample)
        assert np.abs(np.mean(arr)) < 0.1
        assert np.abs(np.std(arr) - 1.0) < 0.1

    def test_draw_uniform_in_range(self) -> None:
        """Verify draw_uniform respects bounds."""
        rng = SeededRNG(seed=42)
        low, high = 5.0, 15.0
        arr = rng.draw_uniform(low, high, (100,))

        assert np.all(arr >= low)
        assert np.all(arr < high)

    def test_draw_uniform_scalar(self) -> None:
        """Verify draw_uniform with size=None returns 0-d array."""
        rng = SeededRNG(seed=42)
        val = rng.draw_uniform(0, 10, size=None)
        assert isinstance(val, np.ndarray)
        assert val.ndim == 0

    def test_draw_choice_from_array(self) -> None:
        """Verify draw_choice samples from provided array."""
        rng = SeededRNG(seed=42)
        choices = np.array([1, 2, 3, 4, 5])
        result = rng.draw_choice(choices, size=10)

        assert result.shape == (10,)
        assert np.all(np.isin(result, choices))

    # -- Failure path: invalid seeds ----------------------------------------

    def test_negative_seed_raises_error(self) -> None:
        """Verify negative seed raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            SeededRNG(seed=-1)

    def test_float_seed_raises_error(self) -> None:
        """Verify non-integer seed raises ValueError."""
        with pytest.raises(ValueError, match="non-negative integer"):
            SeededRNG(seed=3.14)  # type: ignore[arg-type]

    def test_string_seed_raises_error(self) -> None:
        """Verify string seed raises ValueError."""
        with pytest.raises(ValueError, match="non-negative integer"):
            SeededRNG(seed="42")  # type: ignore[arg-type]

    # -- Interface: attribute access ----------------------------------------

    def test_seed_attribute_stores_seed(self) -> None:
        """Verify seed attribute reflects initialization."""
        seed = 999
        rng = SeededRNG(seed=seed)
        assert rng.seed == seed

    def test_generator_attribute_exists(self) -> None:
        """Verify generator attribute is a numpy.random.Generator."""
        rng = SeededRNG(seed=42)
        assert isinstance(rng.generator, np.random.Generator)
