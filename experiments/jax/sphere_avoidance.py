#!/usr/bin/env python3

import time
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tbai_cbf_mppi.systems import SimpleSingleIntegrator2D
from tbai_cbf_mppi.cbf import ControlBarrierFunctionFactory, visualize_cbfs
from tbai_cbf_mppi.control import VanillaSafetyFilterNew
from tbai_cbf_mppi.jax import (
  setup_jax,
  jit_expr,
  MppiConfig,
  create_mppi,
  make_mppi_step,
  make_cost_evaluator,
  reset_relaxation,
)
from tbai_cbf_mppi.anim import save_animation


def main(show_animation=True, device="cpu"):
  active_device = setup_jax(enable_x64=True, device=device)
  print(f"JAX device: {active_device}")

  # Initial and final states
  x_initial = np.array([-3.0, -3.4])
  x_desired = np.array([3.0, 3.0])

  # Strict CBFs (rectangles)
  factory1 = ControlBarrierFunctionFactory()
  cbf1 = factory1.get_rectangle(method="approximate").substitute(c_x=3, c_y=-0.5, w=2.0, h=2.0)
  cbf2 = factory1.get_rectangle(method="approximate").substitute(c_x=-3, c_y=-0.5, w=2.0, h=2.0)
  cbf3 = factory1.union([cbf1, cbf2], method="approximate").substitute(kappa=10)

  # Soft CBFs (spheres)
  factory2 = ControlBarrierFunctionFactory()
  cbf4 = factory2.get_sphere().substitute(c_x=1, c_y=0, r=1.3)
  cbf5 = factory2.get_sphere().substitute(c_x=-1, c_y=0.2, r=1.3)

  # Visualize CBFs
  colors = ["red", "green", "blue"]
  fig, ax = plt.subplots()
  ax.set_aspect("equal")
  ax.set_xlim(-4, 4)
  ax.set_ylim(-4, 4)
  pcm = visualize_cbfs([cbf3, cbf4, cbf5], ax, granularity=200, unsafe_colors=colors, alpha=0.5)

  # Visualize initial and desired states
  ax.plot(x_initial[0], x_initial[1], "rx", label="Initial state", markersize=10)
  ax.plot(x_desired[0], x_desired[1], "bx", label="Desired state", markersize=10)
  ax.legend()

  # Initialize system
  system = SimpleSingleIntegrator2D().reset(x_initial, visualizer=(fig, ax), visualize_history=True)
  system.visualize()
  dt = 0.02

  # Get LQR controller
  Q, R = np.eye(2), np.eye(2)

  # Get safety filter
  safety_filter = VanillaSafetyFilterNew(system.get_A(), system.get_B(), cbf3, alpha=5.5)

  # Prepare stage cost expressions
  x1, x2, u1, u2 = factory1.x, factory1.y, factory1.u1, factory1.u2
  lqr_stage_cost_expr = system.get_lqr_cost_expr(Q, R, x1, x2, u1, u2, x_desired[0], x_desired[1], 0.0, 0.0)
  lqr_stage_jit = jit_expr(lqr_stage_cost_expr)
  cbf4_jit = jit_expr(cbf4.get_expr(substitute=True))
  cbf5_jit = jit_expr(cbf5.get_expr(substitute=True))

  lqr_final_cost_expr = system.get_lqr_cost_expr(Q, R, x1, x2, u1, u2, x_desired[0], x_desired[1], 0.0, 0.0)
  lqr_final_jit = jit_expr(lqr_final_cost_expr)

  # Plain JAX cost functions (no @cost_fn decorator needed)
  def stage_cost(x, y, u1, u2, weight1, weight2, alpha):
    cost = alpha * lqr_stage_jit(x, y, u1, u2)
    cbf4_val = cbf4_jit(x, y)
    cbf5_val = cbf5_jit(x, y)
    cost = cost + jnp.where(cbf4_val < 0, -weight1 * cbf4_val, 0.0)
    cost = cost + jnp.where(cbf5_val < 0, -weight2 * cbf5_val, 0.0)
    return cost

  def terminal_cost(x, y, weight1, weight2, alpha):
    return lqr_final_jit(x, y, 0.0, 0.0)

  cost_evaluator = make_cost_evaluator(stage_cost, terminal_cost)

  # Create MPPI
  config = MppiConfig(
    dt=0.02,
    horizon=20,
    mc_rollouts=1000,
    lmbda=50.0,
    sigma=np.eye(2) * 4.0,
    max_abs_velocity=2.3,
    transition_time=20,
  )
  config, mppi_state = create_mppi(
    system=system,
    config=config,
    lqr_Q=Q,
    lqr_R=R,
    x_desired=x_desired,
    rng_seed=42,
  )

  # Create JIT-compiled step function (compiles entire MPPI loop into one XLA program)
  mppi_step = make_mppi_step(config, cost_evaluator, return_optimal_trajectory=True)

  # JIT warmup
  print("Warming up JIT...")
  warmup_args = (
    (1.0, 5000.0),  # scalar_args (weight1, weight2)
    (jnp.array(mppi_state.relaxation_alphas),),  # ew_args
    (),  # vw_args
  )
  _, _, mppi_state, _, _ = mppi_step(mppi_state, jnp.asarray(x_initial), *warmup_args)
  # Reset state after warmup
  _, mppi_state = create_mppi(
    system=system,
    config=config,
    lqr_Q=Q,
    lqr_R=R,
    x_desired=x_desired,
    rng_seed=42,
  )
  print("JIT warmup complete!")

  print("\n" + "=" * 50)
  print("CONTROLS:")
  print("  [c] - Cycle obstacle priorities")
  print("  [x] - Reset simulation")
  print("=" * 50 + "\n")

  (optimal_trajectory_plot,) = ax.plot([], [], "k-", label="Optimal trajectory")
  weight1 = 1.0
  weight2 = 5000.0
  flip = False

  ax.minorticks_on()
  ax.grid(which="both", alpha=0.2)

  def on_key_press(event):
    nonlocal flip, system, fig, ax
    if event.key == "c":
      flip = True
    if event.key == "x":
      system.reset(x_initial, visualizer=(fig, ax), visualize_history=True)

  fig.canvas.mpl_connect("key_press_event", on_key_press)

  current_time = 0.0

  @save_animation(fig, ax, filename="animation_jax.gif", fps=20, repeat=True, include_all=True)
  def update(i):
    nonlocal weight1, weight2, flip, pcm, colors, current_time, mppi_state

    if flip:
      weight1, weight2 = weight2, weight1
      mppi_state = reset_relaxation(mppi_state, config.transition_time, config.horizon)
      colors[1], colors[2] = colors[2], colors[1]
      pcm = visualize_cbfs([cbf3, cbf4, cbf5], ax, granularity=200, unsafe_colors=colors, alpha=0.5, pcm=pcm)
      print("Flipped")
      flip = False

    t1 = time.time()
    x0 = jnp.asarray(system.state)
    control_jax, _, mppi_state, optimal_trajectory, _ = mppi_step(
      mppi_state, x0,
      (weight1, weight2),  # scalar_args
      (mppi_state.relaxation_alphas,),  # ew_args
      (),  # vw_args
    )
    # Device-to-host transfer for CasADi safety filter (CPU)
    control = np.asarray(control_jax)
    control = safety_filter.solve(state=system.state, u_nominal=control)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.4f} seconds")

    system.step(control, dt=dt)
    system.visualize()

    fig.suptitle(f"Time: {current_time:.2f} s")
    current_time += dt
    time.sleep(0.05)

    optimal_trajectory = np.asarray(optimal_trajectory)
    if optimal_trajectory is not None:
      optimal_trajectory_plot.set_data(optimal_trajectory[:, 0], optimal_trajectory[:, 1])

  anim = FuncAnimation(fig, update, interval=33, frames=100)
  if show_animation:
    plt.show()
  else:
    plt.ioff()
    for frame in range(100):
      update(frame)
    plt.ion()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--no_gui", action="store_true")
  parser.add_argument("--cuda", action="store_true", help="Run on GPU")
  args = parser.parse_args()
  main(show_animation=not args.no_gui, device="cuda" if args.cuda else "cpu")
