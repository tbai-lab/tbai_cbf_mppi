#!/usr/bin/env python3

import numpy as np
from tbai_cbf_mppi.cbf import ControlBarrierFunctionFactory
from tbai_cbf_mppi.control import (
  VanillaSafetyFilterNew,
  PDRegulator,
  ProblemType,
  Solver,
  get_default_solver,
  get_default_solver_options,
)
from tbai_cbf_mppi.systems import SimpleSingleIntegrator2D


def test_vanilla_safety_filter_new():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0, r=1)

  system = SimpleSingleIntegrator2D()
  filter = VanillaSafetyFilterNew(system.get_A(), system.get_B(), cbf, alpha=1.0)

  state = np.array([1.0, 1.0])
  u_nominal = np.array([0.0, 0.0])
  u = filter.solve(state=state, u_nominal=u_nominal)

  assert np.allclose(u, u_nominal)

  state = np.array([0.3, 0.0])
  u_nominal = np.array([0.0, 0.0])
  u = filter.solve(state=state, u_nominal=u_nominal)

  # Should be moving away from the sphere
  assert u[0] > 0
  assert np.allclose(u[1], 0.0)


def test_vanilla_safety_filter_new2():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0)

  system = SimpleSingleIntegrator2D()
  filter = VanillaSafetyFilterNew(system.get_A(), system.get_B(), cbf, alpha=1.0)

  state = np.array([1.0, 1.0])
  u_nominal = np.array([0.0, 0.0])
  u = filter.solve(state=state, u_nominal=u_nominal, r_1=1.0)

  assert np.allclose(u, u_nominal)

  state = np.array([0.3, 0.0])
  u_nominal = np.array([0.0, 0.0])
  u = filter.solve(state=state, u_nominal=u_nominal, r_1=1.0)

  # Should be moving away from the sphere
  assert u[0] > 0
  assert np.allclose(u[1], 0.0)

  u = filter.solve(state=state, u_nominal=u_nominal, r_1=0.1)

  # Is outside the sphere
  assert np.allclose(u, u_nominal)


def test_pd_regulator_proportional():
  pd = PDRegulator(kp=1.0, kd=0.0)
  # Pure proportional: output = kp * error
  control = pd.solve_error(np.array([2.0, 3.0]))
  assert np.allclose(control, [2.0, 3.0])


def test_pd_regulator_derivative():
  pd = PDRegulator(kp=0.0, kd=1.0)
  # First call: derivative is 0 (error - last_error = 0 when last_error is None)
  control = pd.solve_error(np.array([1.0, 1.0]), dt=1.0)
  assert np.allclose(control, [0.0, 0.0])

  # Second call: derivative = (2 - 1) / 1 = 1
  control = pd.solve_error(np.array([2.0, 2.0]), dt=1.0)
  assert np.allclose(control, [1.0, 1.0])


def test_pd_regulator_solve():
  pd = PDRegulator(kp=2.0, kd=0.0)
  control = pd.solve(desired=np.array([5.0, 3.0]), current=np.array([1.0, 1.0]))
  assert np.allclose(control, [8.0, 4.0])


def test_pd_regulator_dt():
  pd = PDRegulator(kp=0.0, kd=1.0)
  pd.solve_error(np.array([0.0]), dt=0.1)
  control = pd.solve_error(np.array([1.0]), dt=0.1)
  # derivative = (1 - 0) / 0.1 = 10
  assert np.allclose(control, [10.0])


def test_get_default_solver():
  assert get_default_solver(ProblemType.QP) == Solver.QPOASES
  assert get_default_solver(ProblemType.NLP) == Solver.IPOPT


def test_get_default_solver_options():
  opts = get_default_solver_options(Solver.OSQP)
  assert opts == {}

  opts = get_default_solver_options(Solver.QPOASES)
  assert "printLevel" in opts
  assert opts["printLevel"] == "none"

  opts = get_default_solver_options(Solver.IPOPT)
  assert "ipopt.print_level" in opts


def test_safety_filter_modifies_unsafe_input():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0, r=1)

  system = SimpleSingleIntegrator2D()
  sf = VanillaSafetyFilterNew(system.get_A(), system.get_B(), cbf, alpha=1.0)

  # Near boundary, pushing inward - should be corrected
  state = np.array([1.1, 0.0])
  u_nominal = np.array([-5.0, 0.0])  # pushing into obstacle
  u_safe = sf.solve(state=state, u_nominal=u_nominal)

  # Should not go as far inward
  assert u_safe[0] > u_nominal[0]
