#!/usr/bin/env python3

from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import sympy as sp

from tbai_cbf_mppi.systems import SimpleSingleIntegrator2D
from tbai_cbf_mppi.jax.symperf_jax import jit_expr


def setup_jax(enable_x64=True, device=None):
  """Configure JAX precision and default device.

  Call this before creating any JAX arrays.

  Args:
    enable_x64: Enable float64 support (default True).
    device: Device to use: "cpu", "gpu"/"cuda", or None for auto-detect.

  Returns:
    The configured jax.Device.
  """
  if enable_x64:
    jax.config.update("jax_enable_x64", True)
  if device is not None:
    if device in ("gpu", "cuda"):
      gpus = jax.devices("gpu")
      if not gpus:
        raise RuntimeError(
          "No GPU available. Install jax[cuda12] or jax[cuda13]."
        )
      dev = gpus[0]
      jax.config.update("jax_default_device", dev)
      return dev
    elif device == "cpu":
      dev = jax.devices("cpu")[0]
      jax.config.update("jax_default_device", dev)
      return dev
    else:
      raise ValueError(f"Unknown device: {device}. Use 'cpu', 'gpu', or 'cuda'.")
  return default_device()


def has_gpu():
  """Check if a GPU backend is available."""
  try:
    return len(jax.devices("gpu")) > 0
  except RuntimeError:
    return False


def default_device():
  """Return the current default JAX device."""
  return jnp.zeros(0).devices().pop()


class MppiConfig(NamedTuple):
  dt: float = 0.02
  horizon: int = 100
  mc_rollouts: int = 1000
  lmbda: float = 50.0
  sigma: np.ndarray = np.eye(2) * 2.0
  max_abs_velocity: float = 2.3
  transition_time: int = 3


class MppiState(NamedTuple):
  u_prev: jnp.ndarray  # (T, 2)
  Ahat: jnp.ndarray  # (2T, 2)
  Bhat: jnp.ndarray  # (2T, 2T)
  lqr_matrix: jnp.ndarray  # (2, 2)
  x_desired: jnp.ndarray  # (2,)
  relaxation_alphas: jnp.ndarray  # (T,)
  rng_key: jnp.ndarray


def discretize_system(A, B, dt):
  """Discretize a linear system using Euler integration.

  Args:
    A: The system matrix of shape (n, n)
    B: The control matrix of shape (n, m)
    dt: The time step

  Returns:
    Ad: The discretized system matrix of shape (n, n)
    Bd: The discretized control matrix of shape (n, m)
  """
  assert A.shape[0] == A.shape[1], "A must be a square matrix"
  assert B.shape[0] == A.shape[0], "B must have the same number of rows as A"
  Ad = np.eye(A.shape[0]) + A * dt
  Bd = B * dt
  return Ad, Bd


def get_hat_system(A: np.ndarray, B: np.ndarray, T: int, dtype=np.float32):
  """Get the hat system for the given system.

  Args:
    A: The system matrix of shape (2, 2)
    B: The control matrix of shape (2, 2)
    T: The number of time steps
    dtype: The data type of the matrices

  Returns:
    Ahat: The hat system matrix of shape (2 * T, 2)
    Bhat: The hat control matrix of shape (2 * T, 2 * T)
  """
  assert A.shape == (2, 2) and B.shape == (2, 2), (
    f"A and B must be 2x2 and 2x2 matrices, found: {A.shape} and {B.shape}"
  )
  Bhat = np.zeros((2 * T, 2 * T))
  Ahat = np.zeros((2 * T, 2))
  for i in range(T):
    Ahat[2 * i : 2 * i + 2, :] = Ahat[2 * (i - 1) : 2 * i, :] @ A if i > 0 else np.eye(2)

  for j in range(0, T):
    Bhat[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = B
    for i in range(j + 1, T):
      Bhat[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = A @ Bhat[2 * (i - 1) : 2 * i, 2 * j : 2 * j + 2]
  return Ahat.astype(dtype), Bhat.astype(dtype)


def create_mppi(
  system: SimpleSingleIntegrator2D,
  config: MppiConfig,
  lqr_Q: np.ndarray = np.eye(2),
  lqr_R: np.ndarray = np.eye(2),
  x_desired: np.ndarray = np.array([2.0, 2.0]),
  rng_seed: int = 0,
  dtype=np.float64,
):
  """Create MPPI config and initial state.

  Returns:
    (MppiConfig, MppiState)
  """
  lqr_matrix = jnp.array(system.get_lqr_gain(lqr_Q, lqr_R), dtype=dtype)

  A, B = system.get_A(), system.get_B()
  Ad, Bd = discretize_system(A, B, config.dt)
  Ahat, Bhat = get_hat_system(Ad, Bd, config.horizon, dtype=dtype)
  Ahat = jnp.array(Ahat, dtype=dtype)
  Bhat = jnp.array(Bhat, dtype=dtype)

  u_prev = jnp.zeros((config.horizon, system.dim_u), dtype=dtype)
  x_desired_arr = jnp.array(x_desired, dtype=dtype)
  relaxation_alphas = jnp.ones(config.horizon, dtype=dtype)

  rng_key = jax.random.PRNGKey(rng_seed)

  state = MppiState(
    u_prev=u_prev,
    Ahat=Ahat,
    Bhat=Bhat,
    lqr_matrix=lqr_matrix,
    x_desired=x_desired_arr,
    relaxation_alphas=relaxation_alphas,
    rng_key=rng_key,
  )
  return config, state


def compute_weights(S, lmbda):
  """Compute MPPI importance weights from costs."""
  rho = jnp.min(S)
  exp_terms = jnp.exp((-1.0 / lmbda) * (S - rho))
  eta = jnp.sum(exp_terms)
  w = (1.0 / eta) * exp_terms
  return w


def moving_average_filter(xx, window_size):
  """Apply a moving average filter with boundary correction.

  Args:
    xx: Input array of shape (T, D)
    window_size: Size of the convolution window

  Returns:
    Filtered array of shape (T, D)
  """
  kernel = jnp.ones(window_size) / window_size

  def convolve_col(col):
    return jnp.convolve(col, kernel, mode="same")

  xx_mean = jax.vmap(convolve_col, in_axes=1, out_axes=1)(xx)

  n_conv = int(np.ceil(window_size / 2))
  xx_mean = xx_mean.at[0, :].multiply(window_size / n_conv)

  idx = jnp.arange(1, n_conv)
  scale_start = window_size / (idx + n_conv)
  scale_end = window_size / (idx + n_conv - (window_size % 2))

  xx_mean = xx_mean.at[idx, :].multiply(scale_start[:, None])
  xx_mean = xx_mean.at[-idx, :].multiply(scale_end[:, None])

  return xx_mean


def make_cost_evaluator(stage_cost_fn, terminal_cost_fn):
  """Create a JIT-compiled cost evaluator from stage and terminal cost functions.

  stage_cost_fn(x, y, u1, u2, *scalar_args, *ew_args_t, *vw_args) -> scalar
  terminal_cost_fn(x, y, *scalar_args, *ew_args_t, *vw_args) -> scalar

  Returns:
    evaluate(X, U, scalar_args, ew_args, vw_args) -> S of shape (K,)
    where:
      X: (K, T, 2) states
      U: (K, T, 2) controls
      scalar_args: tuple of scalars (constant per rollout per timestep)
      ew_args: tuple of arrays each (T,) (element-wise per timestep)
      vw_args: tuple of arrays (vector-wise, constant per timestep)
  """

  def single_rollout_cost(x_traj, u_traj, scalar_args, ew_args, vw_args):
    """Cost for a single rollout: x_traj (T,2), u_traj (T,2)."""
    T = x_traj.shape[0]

    def step_cost(carry, t):
      ew_vals = tuple(a[t] for a in ew_args)
      c = stage_cost_fn(x_traj[t, 0], x_traj[t, 1], u_traj[t, 0], u_traj[t, 1], *scalar_args, *ew_vals, *vw_args)
      return carry + c, None

    total, _ = jax.lax.scan(step_cost, 0.0, jnp.arange(T))

    # Terminal cost at last timestep
    ew_vals_last = tuple(a[T - 1] for a in ew_args)
    total = total + terminal_cost_fn(x_traj[T - 1, 0], x_traj[T - 1, 1], *scalar_args, *ew_vals_last, *vw_args)
    return total

  def evaluate(X, U, scalar_args=(), ew_args=(), vw_args=()):
    """Evaluate cost over all rollouts.

    Args:
      X: (K, T, 2) trajectories
      U: (K, T, 2) controls
      scalar_args: tuple of scalars
      ew_args: tuple of (T,) arrays
      vw_args: tuple of arrays

    Returns:
      S: (K,) cost per rollout
    """
    batch_fn = jax.vmap(single_rollout_cost, in_axes=(0, 0, None, None, None))
    return batch_fn(X, U, scalar_args, ew_args, vw_args)

  return jax.jit(evaluate)


def reset_relaxation(state, transition_time, horizon, init=False):
  """Reset relaxation alphas in the state."""
  if init:
    alphas = jnp.ones(horizon)
  else:
    ramp = jnp.linspace(0, 1, transition_time)
    ones = jnp.ones(horizon - transition_time)
    alphas = jnp.concatenate([ramp, ones])
  return state._replace(relaxation_alphas=alphas)


def _step_relaxation(relaxation_alphas):
  """Shift relaxation window: drop first, append 1."""
  return jnp.concatenate([relaxation_alphas[1:], jnp.array([1.0])])


def make_mppi_step(config, cost_evaluator, return_optimal_trajectory=False, return_sampled_trajectories=False):
  """Create a JIT-compiled MPPI step function.

  Closes over config (static) and the cost evaluator function, compiling the
  entire MPPI iteration into a single XLA program.

  Args:
    config: MppiConfig (closed over, static)
    cost_evaluator: A single cost evaluator from make_cost_evaluator. To use
      multiple evaluators, compose them: ``lambda *a: eval1(*a) + eval2(*a)``.
    return_optimal_trajectory: Whether to compute optimal trajectory (static)
    return_sampled_trajectories: Whether to compute sampled trajectories (static)

  Returns:
    step(state, x0, scalar_args, ew_args, vw_args) -> (control, u_seq, new_state, optimal_traj, sampled_trajs)
    All outputs are jax arrays. optimal_traj/sampled_trajs are zeros when disabled.
  """
  T = config.horizon
  K = config.mc_rollouts
  dt = config.dt
  lmbda = config.lmbda
  max_v = config.max_abs_velocity
  sigma = jnp.array(config.sigma)

  @jax.jit
  def step(state, x0, scalar_args=(), ew_args=(), vw_args=()):
    # LQR warmstart via scan
    u_prev = state.u_prev
    a = 0.3 * state.relaxation_alphas[0]

    def lqr_warmstart_step(carry, i):
      current_state, u_seq = carry
      u_lqr = -state.lqr_matrix @ (current_state - state.x_desired)
      u_new = u_lqr * a + (1 - a) * u_seq[i]
      next_state = current_state + u_new * dt
      u_seq_new = u_seq.at[i].set(u_new)
      return (next_state, u_seq_new), None

    (_, u_prev), _ = jax.lax.scan(lqr_warmstart_step, (x0, u_prev), jnp.arange(T))
    u = u_prev

    # Sample noise
    rng_key, subkey = jax.random.split(state.rng_key)
    epsilon = jax.random.multivariate_normal(subkey, jnp.zeros(2), sigma, shape=(K, T))

    # Perturbed controls
    v = jnp.clip(u[None, :, :] + epsilon, -max_v, max_v)

    # Trajectory rollout via Ahat/Bhat
    v_flat = v.reshape(K, T * 2)
    out = (
      jnp.matmul(state.Bhat[None, :, :], v_flat[:, :, None]).squeeze(-1)
      + jnp.matmul(state.Ahat, x0)[None, :]
    ).reshape(K, T, 2)

    # Cost
    S = cost_evaluator(out, v, scalar_args, ew_args, vw_args)

    # Weights and weighted noise
    w = compute_weights(S, lmbda)
    w_epsilon = jnp.sum(w[:, None, None] * epsilon, axis=0)

    # Moving average filter
    w_epsilon = moving_average_filter(xx=w_epsilon, window_size=T)
    u = u + w_epsilon * jnp.maximum(state.relaxation_alphas[0], 0.1)

    # Step relaxation
    new_alphas = _step_relaxation(state.relaxation_alphas)

    # Optional optimal trajectory
    if return_optimal_trajectory:
      def integrate_step(x, t):
        u_t = jnp.clip(u[t], -max_v, max_v)
        return x + u_t * dt, x + u_t * dt

      _, optimal_traj = jax.lax.scan(integrate_step, x0, jnp.arange(T))
    else:
      optimal_traj = jnp.zeros((T, 2))

    # Optional sampled trajectories
    if return_sampled_trajectories:
      sorted_idx = jnp.argsort(S)

      def rollout_one(k):
        def step_fn(x, t):
          v_t = jnp.clip(v[k, t], -max_v, max_v)
          return x + v_t * dt, x + v_t * dt

        _, traj = jax.lax.scan(step_fn, x0, jnp.arange(T))
        return traj

      sampled_trajs = jax.vmap(rollout_one)(sorted_idx)
    else:
      sampled_trajs = jnp.zeros((K, T, 2))

    # Shift u_prev
    new_u_prev = jnp.concatenate([u[1:], u[-1:]], axis=0)

    new_state = state._replace(
      u_prev=new_u_prev,
      relaxation_alphas=new_alphas,
      rng_key=rng_key,
    )

    return u[0], u, new_state, optimal_traj, sampled_trajs

  return step


def calc_control_input(
  config: MppiConfig,
  state: MppiState,
  observed_x: np.ndarray,
  cost_evaluators,
  x_desired: np.ndarray = None,
  return_optimal_trajectory: bool = False,
  return_sampled_trajectories: bool = False,
):
  """Compute control input using MPPI (convenience wrapper, not fully JIT'd).

  For a fully JIT-compiled version, use make_mppi_step() instead.

  Args:
    config: MppiConfig
    state: MppiState
    observed_x: Current observed state (2,)
    cost_evaluators: List of (evaluator_fn, (scalar_args, ew_args, vw_args)) tuples
    x_desired: Override desired state (optional)
    return_optimal_trajectory: Whether to compute and return optimal trajectory
    return_sampled_trajectories: Whether to compute and return sampled trajectories

  Returns:
    (control, u_seq, new_state, optimal_traj, sampled_trajs)
  """
  T = config.horizon
  K = config.mc_rollouts

  x0 = jnp.asarray(observed_x)
  xd = jnp.asarray(x_desired) if x_desired is not None else state.x_desired
  state = state._replace(x_desired=xd)

  # LQR warmstart via scan
  u_prev = state.u_prev
  a = 0.3 * state.relaxation_alphas[0]

  def lqr_warmstart_step(carry, i):
    current_state, u_prev_seq = carry
    u_lqr = -state.lqr_matrix @ (current_state - xd)
    u_new = u_lqr * a + (1 - a) * u_prev_seq[i]
    next_state = current_state + u_new * config.dt
    u_prev_new = u_prev_seq.at[i].set(u_new)
    return (next_state, u_prev_new), None

  (_, u_prev), _ = jax.lax.scan(lqr_warmstart_step, (x0, u_prev), jnp.arange(T))
  u = u_prev

  # Sample noise
  rng_key, subkey = jax.random.split(state.rng_key)
  mu = jnp.zeros(2)
  sigma = jnp.array(config.sigma)
  epsilon = jax.random.multivariate_normal(subkey, mu, sigma, shape=(K, T))

  # Perturbed controls
  v = jnp.clip(u[None, :, :] + epsilon, -config.max_abs_velocity, config.max_abs_velocity)

  # Trajectory rollout via Ahat/Bhat
  v_flat = v.reshape(K, T * 2)
  out = (
    jnp.matmul(state.Bhat[None, :, :], v_flat[:, :, None]).squeeze(-1) + jnp.matmul(state.Ahat, x0)[None, :]
  ).reshape(K, T, 2)

  # Cost accumulation
  S = jnp.zeros(K)
  for evaluator_fn, args in cost_evaluators:
    scalar_args, ew_args, vw_args = args
    S = S + evaluator_fn(out, v, scalar_args, ew_args, vw_args)

  # Compute weights
  w = compute_weights(S, config.lmbda)
  w_epsilon = jnp.sum(w[:, None, None] * epsilon, axis=0)

  # Moving average filter
  w_epsilon = moving_average_filter(xx=w_epsilon, window_size=T)
  u = u + w_epsilon * jnp.maximum(state.relaxation_alphas[0], 0.1)

  # Step relaxation
  new_alphas = _step_relaxation(state.relaxation_alphas)

  # Compute optional trajectories
  optimal_traj = None
  if return_optimal_trajectory:
    def integrate_step(x, t):
      u_t = jnp.clip(u[t], -config.max_abs_velocity, config.max_abs_velocity)
      x_next = x + u_t * config.dt
      return x_next, x_next

    _, optimal_traj = jax.lax.scan(integrate_step, x0, jnp.arange(T))

  sampled_trajs = None
  if return_sampled_trajectories:
    sorted_idx = jnp.argsort(S)

    def rollout_one(k):
      def step(x, t):
        v_t = jnp.clip(v[k, t], -config.max_abs_velocity, config.max_abs_velocity)
        x_next = x + v_t * config.dt
        return x_next, x_next

      _, traj = jax.lax.scan(step, x0, jnp.arange(T))
      return traj

    sampled_trajs = jax.vmap(rollout_one)(sorted_idx)

  # Shift u_prev for next iteration
  new_u_prev = jnp.concatenate([u[1:], u[-1:]], axis=0)

  new_state = state._replace(
    u_prev=new_u_prev,
    relaxation_alphas=new_alphas,
    rng_key=rng_key,
  )

  control = np.array(u[0])
  u_seq = np.array(u)
  optimal_traj_np = np.array(optimal_traj) if optimal_traj is not None else None
  sampled_trajs_np = np.array(sampled_trajs) if sampled_trajs is not None else None

  return control, u_seq, new_state, optimal_traj_np, sampled_trajs_np


def jit_expr_v2t(expr: sp.Expr, symbols: list[str]):
  """Jit an expression that takes x, y and a val array.

  Returns a function fn(x, y, val_array) that unpacks val_array entries
  as the additional symbols.
  """
  jitted = jit_expr(expr, symbols=["x", "y"] + symbols)

  @jax.jit
  def wrapper(x, y, val):
    args = [x, y] + [val[i] for i in range(len(symbols))]
    return jitted(*args)

  return wrapper
