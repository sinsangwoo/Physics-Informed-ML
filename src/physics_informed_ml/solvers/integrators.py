"""Numerical integrators for physics simulations."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch


class Integrator(ABC):
    """Abstract base class for numerical integrators."""

    @abstractmethod
    def step(
        self, state: np.ndarray, derivative_fn: Callable, dt: float
    ) -> np.ndarray:
        """Perform one integration step.

        Args:
            state: Current state vector
            derivative_fn: Function computing derivative given state
            dt: Time step

        Returns:
            Next state vector
        """
        pass


class EulerIntegrator(Integrator):
    """Simple Euler integration (first-order).

    x_{n+1} = x_n + dt * f(x_n)
    """

    def step(
        self, state: np.ndarray, derivative_fn: Callable, dt: float
    ) -> np.ndarray:
        """Euler integration step.

        Args:
            state: Current state
            derivative_fn: Derivative function
            dt: Time step

        Returns:
            Next state
        """
        return state + dt * derivative_fn(state)


class RK4Integrator(Integrator):
    """Fourth-order Runge-Kutta integration.

    Classic RK4 method with O(dt^4) accuracy.
    """

    def step(
        self, state: np.ndarray, derivative_fn: Callable, dt: float
    ) -> np.ndarray:
        """RK4 integration step.

        Args:
            state: Current state
            derivative_fn: Derivative function
            dt: Time step

        Returns:
            Next state
        """
        k1 = derivative_fn(state)
        k2 = derivative_fn(state + 0.5 * dt * k1)
        k3 = derivative_fn(state + 0.5 * dt * k2)
        k4 = derivative_fn(state + dt * k3)

        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class VerletIntegrator(Integrator):
    """Velocity Verlet integration (symplectic, energy-conserving).

    Particularly good for Hamiltonian systems like pendulums.
    """

    def step(
        self, state: np.ndarray, derivative_fn: Callable, dt: float
    ) -> np.ndarray:
        """Verlet integration step.

        Note: For second-order ODEs, derivative_fn should return acceleration.

        Args:
            state: Current state [position, velocity]
            derivative_fn: Returns [velocity, acceleration]
            dt: Time step

        Returns:
            Next state [position, velocity]
        """
        # Split state into position and velocity
        n = len(state) // 2
        q, v = state[:n], state[n:]

        # Get current acceleration
        dstate = derivative_fn(state)
        a = dstate[n:]  # Acceleration part

        # Update position
        q_new = q + v * dt + 0.5 * a * dt**2

        # Get new acceleration
        state_new = np.concatenate([q_new, v])
        dstate_new = derivative_fn(state_new)
        a_new = dstate_new[n:]

        # Update velocity
        v_new = v + 0.5 * (a + a_new) * dt

        return np.concatenate([q_new, v_new])