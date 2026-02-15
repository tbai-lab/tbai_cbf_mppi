#!/usr/bin/env python3

import numpy as np
import pytest
from tbai_cbf_mppi.mppi import (
  discretize_system,
  get_hat_system,
  AcceleratedSafetyMPPI,
  MppiConfig,
  set_default_backend,
  get_default_backend,
  set_default_dtype,
  get_default_dtype,
  set_default_threads_per_block,
  get_default_threads_per_block,
)
from tbai_cbf_mppi.systems import SimpleSingleIntegrator2D


def test_discretize_system_identity():
  A = np.zeros((2, 2))
  B = np.eye(2)
  dt = 0.1
  Ad, Bd = discretize_system(A, B, dt)

  # Ad = I + A*dt = I (since A=0)
  assert np.allclose(Ad, np.eye(2))
  # Bd = B*dt
  assert np.allclose(Bd, np.eye(2) * dt)


def test_discretize_system_nonzero_A():
  A = np.array([[0, 1], [0, 0]])
  B = np.array([[0], [1]])
  dt = 0.01
  Ad, Bd = discretize_system(A, B, dt)

  expected_Ad = np.eye(2) + A * dt
  expected_Bd = B * dt
  assert np.allclose(Ad, expected_Ad)
  assert np.allclose(Bd, expected_Bd)


def test_discretize_system_asserts():
  with pytest.raises(AssertionError):
    discretize_system(np.ones((2, 3)), np.eye(2), 0.1)  # non-square A
  with pytest.raises(AssertionError):
    discretize_system(np.eye(2), np.ones((3, 2)), 0.1)  # B rows != A rows


def test_get_hat_system_shapes():
  A = np.zeros((2, 2))
  B = np.eye(2)
  T = 5
  Ahat, Bhat = get_hat_system(A, B, T)

  assert Ahat.shape == (2 * T, 2)
  assert Bhat.shape == (2 * T, 2 * T)


def test_get_hat_system_single_integrator():
  # For a single integrator (A=0, B=I), Ad=I, Bd=dt*I
  A = np.eye(2)  # discrete A = I
  B = np.eye(2)  # discrete B = I
  T = 3
  Ahat, Bhat = get_hat_system(A, B, T)

  # Ahat first block should be I, second A, third A^2
  assert np.allclose(Ahat[0:2, :], np.eye(2), atol=1e-6)
  assert np.allclose(Ahat[2:4, :], A, atol=1e-6)

  # Bhat should be lower triangular in 2x2 blocks
  # B[0,0] = B, B[1,0] = AB, B[1,1] = B, etc.
  assert np.allclose(Bhat[0:2, 0:2], B, atol=1e-6)
  assert np.allclose(Bhat[2:4, 2:4], B, atol=1e-6)


def test_get_hat_system_asserts():
  with pytest.raises(AssertionError):
    get_hat_system(np.ones((3, 3)), np.eye(2), 5)  # not 2x2


def test_mppi_config_defaults():
  config = MppiConfig()
  assert config.backend == "numpy"
  assert config.dtype in ["float64", "float32", "float16"]


def test_mppi_config_invalid_backend():
  with pytest.raises(AssertionError):
    MppiConfig(backend="invalid")


def test_mppi_config_invalid_dtype():
  with pytest.raises(AssertionError):
    MppiConfig(dtype="int32")


def test_mppi_config_float16_requires_cupy():
  with pytest.raises(AssertionError):
    MppiConfig(backend="numpy", dtype="float16")


def test_default_backend_getset():
  original = get_default_backend()
  set_default_backend("numpy")
  assert get_default_backend() == "numpy"
  set_default_backend(original)


def test_default_dtype_getset():
  original = get_default_dtype()
  set_default_dtype("float32")
  assert get_default_dtype() == "float32"
  set_default_dtype(original)


def test_default_threads_per_block_getset():
  original = get_default_threads_per_block()
  set_default_threads_per_block(128)
  assert get_default_threads_per_block() == 128
  set_default_threads_per_block(original)


def test_set_default_backend_invalid():
  with pytest.raises(AssertionError):
    set_default_backend("tensorflow")


def test_set_default_dtype_invalid():
  with pytest.raises(AssertionError):
    set_default_dtype("int8")


def test_set_default_threads_per_block_invalid():
  with pytest.raises(AssertionError):
    set_default_threads_per_block(-1)


def test_mppi_compute_weights():
  system = SimpleSingleIntegrator2D()
  mppi = AcceleratedSafetyMPPI(system, mc_rollouts=10, horizon=5)

  costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  weights = mppi.compute_weights(costs)

  # Weights should sum to 1
  assert abs(np.sum(weights) - 1.0) < 1e-10
  # Lower cost should get higher weight
  assert weights[0] > weights[-1]
  # All weights should be non-negative
  assert np.all(weights >= 0)


def test_mppi_clamp_input():
  system = SimpleSingleIntegrator2D()
  mppi = AcceleratedSafetyMPPI(system, mc_rollouts=10, horizon=5)

  v = np.array([10.0, -10.0])
  clamped = mppi.clamp_input(v)
  assert np.all(np.abs(clamped) <= mppi.max_abs_velocity)


def test_mppi_moving_average_filter():
  system = SimpleSingleIntegrator2D()
  mppi = AcceleratedSafetyMPPI(system, mc_rollouts=10, horizon=10)

  xx = np.random.randn(10, 2)
  filtered = mppi.moving_average_filter(xx, window_size=3)
  assert filtered.shape == xx.shape


def test_mppi_relaxation():
  system = SimpleSingleIntegrator2D()
  mppi = AcceleratedSafetyMPPI(system, mc_rollouts=10, horizon=5, transition_time=3)

  # Initially all alphas should be 1 (init=True)
  assert all(a == 1 for a in mppi.relaxation_alphas)

  # After reset (init=False), should ramp from 0 to 1
  mppi.reset_relaxation(init=False)
  assert mppi.relaxation_alphas[0] == 0.0
  assert mppi.relaxation_alphas[-1] == 1.0

  # Step should shift the window
  mppi.step_relaxation()
  assert mppi.relaxation_alphas[-1] == 1


def test_mppi_integrate():
  system = SimpleSingleIntegrator2D()
  mppi = AcceleratedSafetyMPPI(system, mc_rollouts=10, horizon=5, dt=0.1)

  x = np.array([0.0, 0.0])
  u = np.array([1.0, 2.0])
  x_next = mppi.integrate(x, u)

  expected = system.integrate(x, u, 0.1)
  assert np.allclose(x_next, expected)
