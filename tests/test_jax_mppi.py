#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tbai_cbf_mppi.jax.mppi_jax import (
  setup_jax,
  has_gpu,
  default_device,
  MppiConfig,
  MppiState,
  create_mppi,
  calc_control_input,
  make_mppi_step,
  compute_weights,
  moving_average_filter,
  make_cost_evaluator,
  discretize_system,
  get_hat_system,
  reset_relaxation,
  jit_expr_v2t,
)
from tbai_cbf_mppi.systems import SimpleSingleIntegrator2D
import sympy as sp

requires_gpu = pytest.mark.skipif(not has_gpu(), reason="No GPU available")


def test_setup_jax_cpu():
  dev = setup_jax(enable_x64=True, device="cpu")
  assert "cpu" in str(dev).lower()


def test_setup_jax_invalid_device():
  with pytest.raises(ValueError, match="Unknown device"):
    setup_jax(device="tpu")


def test_has_gpu_returns_bool():
  assert isinstance(has_gpu(), bool)


def test_default_device_returns_device():
  dev = default_device()
  assert dev is not None


@requires_gpu
def test_setup_jax_gpu():
  dev = setup_jax(enable_x64=True, device="gpu")
  assert "gpu" in str(dev).lower() or "cuda" in str(dev).lower()


@requires_gpu
def test_create_mppi_on_gpu():
  setup_jax(enable_x64=True, device="gpu")
  system = SimpleSingleIntegrator2D()
  config = MppiConfig(dt=0.02, horizon=10, mc_rollouts=50, sigma=np.eye(2) * 4.0)
  config, state = create_mppi(system, config, rng_seed=42)

  # All state arrays should be on GPU
  assert "gpu" in str(state.u_prev.devices()).lower() or "cuda" in str(state.u_prev.devices()).lower()
  assert state.u_prev.shape == (10, 2)


@requires_gpu
def test_make_mppi_step_on_gpu():
  setup_jax(enable_x64=True, device="gpu")
  system = SimpleSingleIntegrator2D()
  config = MppiConfig(dt=0.02, horizon=10, mc_rollouts=50, sigma=np.eye(2) * 4.0)
  config, state = create_mppi(system, config, rng_seed=42)

  def stage_cost(x, y, u1, u2):
    return x**2 + y**2 + u1**2 + u2**2

  def terminal_cost(x, y):
    return x**2 + y**2

  evaluator = make_cost_evaluator(stage_cost, terminal_cost)
  step = make_mppi_step(config, evaluator, return_optimal_trajectory=True)

  x0 = jnp.array([0.0, 0.0])
  control, u_seq, new_state, opt_traj, _ = step(state, x0)

  assert control.shape == (2,)
  assert u_seq.shape == (10, 2)

  # Verify outputs are on GPU
  assert "gpu" in str(control.devices()).lower() or "cuda" in str(control.devices()).lower()

  # Verify device-to-host transfer works
  control_np = np.asarray(control)
  assert isinstance(control_np, np.ndarray)
  assert control_np.shape == (2,)

  # Reset to CPU for other tests
  setup_jax(device="cpu")


def test_discretize_system_identity():
  A = np.zeros((2, 2))
  B = np.eye(2)
  dt = 0.1
  Ad, Bd = discretize_system(A, B, dt)

  assert np.allclose(Ad, np.eye(2))
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
    discretize_system(np.ones((2, 3)), np.eye(2), 0.1)
  with pytest.raises(AssertionError):
    discretize_system(np.eye(2), np.ones((3, 2)), 0.1)


def test_get_hat_system_shapes():
  A = np.zeros((2, 2))
  B = np.eye(2)
  T = 5
  Ahat, Bhat = get_hat_system(A, B, T)

  assert Ahat.shape == (2 * T, 2)
  assert Bhat.shape == (2 * T, 2 * T)


def test_get_hat_system_single_integrator():
  A = np.eye(2)
  B = np.eye(2)
  T = 3
  Ahat, Bhat = get_hat_system(A, B, T)

  assert np.allclose(Ahat[0:2, :], np.eye(2), atol=1e-6)
  assert np.allclose(Ahat[2:4, :], A, atol=1e-6)
  assert np.allclose(Bhat[0:2, 0:2], B, atol=1e-6)
  assert np.allclose(Bhat[2:4, 2:4], B, atol=1e-6)


def test_get_hat_system_asserts():
  with pytest.raises(AssertionError):
    get_hat_system(np.ones((3, 3)), np.eye(2), 5)


def test_compute_weights_sum_to_one():
  S = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
  w = compute_weights(S, lmbda=50.0)
  assert abs(float(jnp.sum(w)) - 1.0) < 1e-6


def test_compute_weights_monotonic():
  S = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
  w = compute_weights(S, lmbda=50.0)
  # Lower cost should get higher weight
  assert float(w[0]) > float(w[-1])


def test_compute_weights_uniform():
  S = jnp.array([1.0, 1.0, 1.0, 1.0])
  w = compute_weights(S, lmbda=50.0)
  assert jnp.allclose(w, jnp.ones(4) / 4.0, atol=1e-10)


def test_moving_average_filter_shape():
  xx = jnp.array(np.random.randn(10, 2))
  filtered = moving_average_filter(xx, window_size=3)
  assert filtered.shape == xx.shape


def test_moving_average_filter_constant():
  # A constant signal should pass through unchanged (approximately)
  xx = jnp.ones((10, 2)) * 3.0
  filtered = moving_average_filter(xx, window_size=3)
  assert jnp.allclose(filtered, xx, atol=1e-5)


def test_create_mppi_shapes():
  system = SimpleSingleIntegrator2D()
  config = MppiConfig(dt=0.02, horizon=20, mc_rollouts=100)
  config, state = create_mppi(system, config, rng_seed=42)

  assert state.u_prev.shape == (20, 2)
  assert state.Ahat.shape == (40, 2)
  assert state.Bhat.shape == (40, 40)
  assert state.lqr_matrix.shape == (2, 2)
  assert state.x_desired.shape == (2,)
  assert state.relaxation_alphas.shape == (20,)


def test_make_cost_evaluator_simple():
  def stage_cost(x, y, u1, u2):
    return x**2 + y**2 + u1**2 + u2**2

  def terminal_cost(x, y):
    return x**2 + y**2

  evaluator = make_cost_evaluator(stage_cost, terminal_cost)

  K, T = 5, 3
  X = jnp.ones((K, T, 2))
  U = jnp.zeros((K, T, 2))

  S = evaluator(X, U)
  assert S.shape == (K,)
  # Each rollout: stage = sum of x^2+y^2 = 2 per step, 3 steps = 6, terminal = 2, total = 8
  assert jnp.allclose(S, jnp.full(K, 8.0), atol=1e-5)


def test_make_cost_evaluator_with_args():
  def stage_cost(x, y, u1, u2, weight, alpha_t):
    return weight * (x**2 + y**2) * alpha_t

  def terminal_cost(x, y, weight, alpha_t):
    return weight * (x**2 + y**2)

  evaluator = make_cost_evaluator(stage_cost, terminal_cost)

  K, T = 4, 3
  X = jnp.ones((K, T, 2))
  U = jnp.zeros((K, T, 2))
  scalar_args = (2.0,)  # weight=2
  ew_args = (jnp.array([1.0, 1.0, 1.0]),)  # alpha per timestep

  S = evaluator(X, U, scalar_args, ew_args)
  assert S.shape == (K,)
  # stage: 2*(1+1)*1 = 4 per step, 3 steps = 12, terminal: 2*(1+1) = 4, total = 16
  assert jnp.allclose(S, jnp.full(K, 16.0), atol=1e-5)


def test_calc_control_input_shapes():
  jax.config.update("jax_enable_x64", True)
  system = SimpleSingleIntegrator2D()
  config = MppiConfig(dt=0.02, horizon=10, mc_rollouts=50, sigma=np.eye(2) * 4.0, transition_time=3)
  config, state = create_mppi(
    system, config, x_desired=np.array([3.0, 3.0]), rng_seed=42,
  )

  def stage_cost(x, y, u1, u2):
    return x**2 + y**2 + u1**2 + u2**2

  def terminal_cost(x, y):
    return x**2 + y**2

  evaluator = make_cost_evaluator(stage_cost, terminal_cost)

  observed_x = np.array([0.0, 0.0])
  control, u_seq, new_state, opt_traj, samp_trajs = calc_control_input(
    config, state, observed_x,
    [(evaluator, ((), (), ()))],
    return_optimal_trajectory=True,
  )

  assert control.shape == (2,)
  assert u_seq.shape == (10, 2)
  assert opt_traj.shape == (10, 2)
  assert samp_trajs is None


def test_calc_control_input_reproducible():
  jax.config.update("jax_enable_x64", True)
  system = SimpleSingleIntegrator2D()
  config = MppiConfig(dt=0.02, horizon=10, mc_rollouts=50, sigma=np.eye(2) * 4.0)

  def stage_cost(x, y, u1, u2):
    return x**2 + y**2

  def terminal_cost(x, y):
    return 0.0

  evaluator = make_cost_evaluator(stage_cost, terminal_cost)
  observed_x = np.array([1.0, 1.0])

  # Two runs with the same seed should produce identical results
  _, state1 = create_mppi(system, config, rng_seed=123)
  control1, _, _, _, _ = calc_control_input(config, state1, observed_x, [(evaluator, ((), (), ()))])

  _, state2 = create_mppi(system, config, rng_seed=123)
  control2, _, _, _, _ = calc_control_input(config, state2, observed_x, [(evaluator, ((), (), ()))])

  assert np.allclose(control1, control2)


def test_relaxation_reset():
  system = SimpleSingleIntegrator2D()
  config = MppiConfig(horizon=10, transition_time=5)
  _, state = create_mppi(system, config, rng_seed=0)

  # Initially all alphas should be 1 (init=True by default in create_mppi)
  assert jnp.allclose(state.relaxation_alphas, jnp.ones(10))

  # After reset (init=False), should ramp from 0 to 1
  state = reset_relaxation(state, transition_time=5, horizon=10, init=False)
  assert float(state.relaxation_alphas[0]) == 0.0
  assert float(state.relaxation_alphas[-1]) == 1.0
  assert state.relaxation_alphas.shape == (10,)


def test_jit_expr_v2t():
  x, y, a, b = sp.symbols("x y a b")
  expr = x + y + a * b
  fn = jit_expr_v2t(expr, symbols=["a", "b"])

  result = fn(1.0, 2.0, jnp.array([3.0, 4.0]))
  assert abs(float(result) - (1.0 + 2.0 + 3.0 * 4.0)) < 1e-10


def test_make_mppi_step_shapes():
  jax.config.update("jax_enable_x64", True)
  system = SimpleSingleIntegrator2D()
  config = MppiConfig(dt=0.02, horizon=10, mc_rollouts=50, sigma=np.eye(2) * 4.0, transition_time=3)
  config, state = create_mppi(system, config, x_desired=np.array([3.0, 3.0]), rng_seed=42)

  def stage_cost(x, y, u1, u2):
    return x**2 + y**2 + u1**2 + u2**2

  def terminal_cost(x, y):
    return x**2 + y**2

  evaluator = make_cost_evaluator(stage_cost, terminal_cost)
  step = make_mppi_step(config, evaluator, return_optimal_trajectory=True)

  x0 = jnp.array([0.0, 0.0])
  control, u_seq, new_state, opt_traj, samp_trajs = step(state, x0)

  assert control.shape == (2,)
  assert u_seq.shape == (10, 2)
  assert opt_traj.shape == (10, 2)
  assert isinstance(new_state, MppiState)


def test_make_mppi_step_reproducible():
  jax.config.update("jax_enable_x64", True)
  system = SimpleSingleIntegrator2D()
  config = MppiConfig(dt=0.02, horizon=10, mc_rollouts=50, sigma=np.eye(2) * 4.0)

  def stage_cost(x, y, u1, u2):
    return x**2 + y**2

  def terminal_cost(x, y):
    return 0.0

  evaluator = make_cost_evaluator(stage_cost, terminal_cost)
  step = make_mppi_step(config, evaluator)

  x0 = jnp.array([1.0, 1.0])

  _, state1 = create_mppi(system, config, rng_seed=123)
  control1, _, _, _, _ = step(state1, x0)

  _, state2 = create_mppi(system, config, rng_seed=123)
  control2, _, _, _, _ = step(state2, x0)

  assert jnp.allclose(control1, control2)


def test_make_mppi_step_with_cost_args():
  jax.config.update("jax_enable_x64", True)
  system = SimpleSingleIntegrator2D()
  config = MppiConfig(dt=0.02, horizon=10, mc_rollouts=50, sigma=np.eye(2) * 4.0)

  def stage_cost(x, y, u1, u2, weight, alpha_t):
    return weight * (x**2 + y**2) * alpha_t

  def terminal_cost(x, y, weight, alpha_t):
    return weight * (x**2 + y**2)

  evaluator = make_cost_evaluator(stage_cost, terminal_cost)
  step = make_mppi_step(config, evaluator)

  _, state = create_mppi(system, config, rng_seed=42)
  x0 = jnp.array([1.0, 1.0])

  scalar_args = (2.0,)
  ew_args = (jnp.ones(10),)

  control, _, new_state, _, _ = step(state, x0, scalar_args, ew_args)
  assert control.shape == (2,)
  assert isinstance(new_state, MppiState)


def test_make_mppi_step_matches_calc_control_input():
  """Verify make_mppi_step produces the same results as calc_control_input."""
  jax.config.update("jax_enable_x64", True)
  system = SimpleSingleIntegrator2D()
  config = MppiConfig(dt=0.02, horizon=10, mc_rollouts=50, sigma=np.eye(2) * 4.0)

  def stage_cost(x, y, u1, u2):
    return x**2 + y**2 + u1**2 + u2**2

  def terminal_cost(x, y):
    return x**2 + y**2

  evaluator = make_cost_evaluator(stage_cost, terminal_cost)
  step = make_mppi_step(config, evaluator, return_optimal_trajectory=True)

  observed_x = np.array([1.0, 1.0])
  x0 = jnp.array([1.0, 1.0])

  # Run both with same seed
  _, state1 = create_mppi(system, config, rng_seed=42)
  control1, u1, _, opt1, _ = calc_control_input(
    config, state1, observed_x, [(evaluator, ((), (), ()))], return_optimal_trajectory=True,
  )

  _, state2 = create_mppi(system, config, rng_seed=42)
  control2, u2, _, opt2, _ = step(state2, x0)

  assert np.allclose(control1, np.array(control2), atol=1e-10)
  assert np.allclose(u1, np.array(u2), atol=1e-10)
  assert np.allclose(opt1, np.array(opt2), atol=1e-10)
