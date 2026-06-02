"""Seeded random number generator wrapper for reproducible simulations.

Provides a single-instance seeded numpy.random.Generator for all randomness
in the simulation (agent spawn, noise, stochastic events). Enables seed-based
reproducibility (FR-6 R6.2) and testing (NFR-R2).
"""
from __future__ import annotations

import numpy as np


class SeededRNG:
    """Thread-unsafe seeded random generator wrapper.

    Wraps numpy.random.Generator to provide reproducible randomness
    across agent spawn, forces, and event simulation. Same seed always
    produces the same sequence of draws.

    Attributes:
        seed: The seed value used to initialize the generator.
        generator: The underlying numpy.random.Generator instance.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the RNG with an optional seed.

        Args:
            seed: Integer seed for reproducibility. If None, uses a random
                seed.

        Raises:
            ValueError: If seed is not a non-negative integer.
        """
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None")

        self.seed: int | None = seed
        self.generator: np.random.Generator = np.random.default_rng(seed)

    def draw_floats(self, shape: tuple[int, ...] | int) -> np.ndarray:
        """Draw random floats in [0, 1) with the given shape.

        Args:
            shape: Shape of the output array, or a single integer for 1D.

        Returns:
            Array of random floats in [0, 1).
        """
        return self.generator.random(shape)

    def draw_randn(self, shape: tuple[int, ...] | int) -> np.ndarray:
        """Draw random floats from standard normal distribution.

        Args:
            shape: Shape of the output array, or a single integer for 1D.

        Returns:
            Array of random normal (μ=0, σ=1) values.
        """
        return self.generator.standard_normal(shape)

    def draw_uniform(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...] | int | None = (),
        size: tuple[int, ...] | int | None = None,
    ) -> np.ndarray:
        """Draw random floats in [low, high) with the given shape.

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).
            shape: Shape of the output array, or a single integer for 1D.
            size: Alias for shape (NumPy convention). If provided, overrides
                shape.

        Returns:
            Array of random floats in [low, high).
        """
        # Allow 'size' as alias for 'shape' (NumPy convention)
        output_shape = size if size is not None else shape
        return self.generator.uniform(low, high, output_shape)

    def draw_choice(
        self,
        a: np.ndarray,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Draw samples from array a.

        Args:
            a: Array to sample from.
            size: Shape of the output, or None for a single scalar.

        Returns:
            Array of samples from a.
        """
        return self.generator.choice(a, size=size)
