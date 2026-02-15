#!/usr/bin/env python3

import numpy as np
import sympy as sp
import pytest
from tbai_cbf_mppi.cbf import ControlBarrierFunctionFactory, ControlBarrierFunctionNew, CasadiControlBarrierFunction


def test_cbf_sphere():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0, r=1)

  # Inside = unsafe
  assert cbf.evaluate(x=0, y=0) < 0
  assert cbf.evaluate(x=0.1, y=0.2) < 0

  # Outside = safe
  assert cbf.evaluate(x=1, y=1) > 0

  # Boundary = safe
  assert cbf.evaluate(x=1, y=0) == 0
  assert cbf.evaluate(x=0, y=1) == 0


def test_cbf_rectangle_exact():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_rectangle(method="exact").substitute(c_x=0, c_y=0, w=2, h=2)

  # Inside = unsafe
  assert cbf.evaluate(x=0, y=0) < 0
  assert cbf.evaluate(x=0.1, y=0.2) < 0

  # Outside = safe
  assert cbf.evaluate(x=2, y=2) > 0

  # Boundary = safe
  assert cbf.evaluate(x=1, y=1) == 0
  assert cbf.evaluate(x=0, y=1) == 0


def test_cbf_rectangle_approx():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_rectangle(method="approximate").substitute(c_x=0, c_y=0, w=2, h=2, kappa=10)

  # Inside = unsafe
  assert cbf.evaluate(x=0, y=0) < 0
  assert cbf.evaluate(x=0.1, y=0.2) < 0
  assert cbf.evaluate(x=1, y=1) > 0  # corner rounding

  # Outside = safe
  assert cbf.evaluate(x=2, y=2) > 0


def test_union_exact():
  factory = ControlBarrierFunctionFactory()
  cbf1 = factory.get_sphere().substitute(c_x=0, c_y=0)
  cbf2 = factory.get_sphere().substitute(c_x=2, c_y=0)
  cbf = factory.union([cbf1, cbf2], method="exact")

  for s in cbf1.h_expr.free_symbols | cbf2.h_expr.free_symbols:
    assert s in cbf.h_expr.free_symbols


def test_union_approximate():
  factory = ControlBarrierFunctionFactory()
  cbf1 = factory.get_sphere().substitute(c_x=0, c_y=0)
  cbf2 = factory.get_sphere().substitute(c_x=2, c_y=0)
  cbf = factory.union([cbf1, cbf2], method="approximate")

  for s in cbf1.h_expr.free_symbols | cbf2.h_expr.free_symbols:
    assert s in cbf.h_expr.free_symbols

  assert factory.kappa in cbf.h_expr.free_symbols


def test_intersection_exact():
  factory = ControlBarrierFunctionFactory()
  cbf1 = factory.get_sphere().substitute(c_x=0, c_y=0)
  cbf2 = factory.get_sphere().substitute(c_x=2, c_y=0)
  cbf = factory.intersection([cbf1, cbf2], method="exact")

  for s in cbf1.h_expr.free_symbols | cbf2.h_expr.free_symbols:
    assert s in cbf.h_expr.free_symbols


def test_intersection_approximate():
  factory = ControlBarrierFunctionFactory()
  cbf1 = factory.get_sphere().substitute(c_x=0, c_y=0)
  cbf2 = factory.get_sphere().substitute(c_x=2, c_y=0)
  cbf = factory.intersection([cbf1, cbf2], method="approximate")

  for s in cbf1.h_expr.free_symbols | cbf2.h_expr.free_symbols:
    assert s in cbf.h_expr.free_symbols

  assert factory.kappa in cbf.h_expr.free_symbols


def test_cbf_get_grad():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0, r=1)
  grad = cbf.get_grad()
  # h = x^2 + y^2 - 1, so dh/dx = 2x, dh/dy = 2y
  x, y = factory.x, factory.y
  assert grad[0] == 2 * x
  assert grad[1] == 2 * y


def test_cbf_get_grad_unsub():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0, r=1)
  grad = cbf.get_grad(substitute=False)
  # Without substitution, gradient still contains parameter symbols
  assert len(grad[0].free_symbols) > 1


def test_cbf_shift_xy():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0, r=1)

  # Shift the CBF by (2, 3)
  cbf.shift_xy(2, 3)

  # The new center should be at (2, 3)
  # Inside the shifted sphere
  assert cbf.evaluate(x=2, y=3) < 0
  # On boundary of shifted sphere
  assert cbf.evaluate(x=3, y=3) == 0
  # Outside shifted sphere
  assert cbf.evaluate(x=0, y=0) > 0


def test_cbf_rotate():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_rectangle(method="exact").substitute(c_x=0, c_y=0, w=4, h=2)

  # Before rotation: point (1.5, 0) is inside (width=4, so half-width=2)
  assert cbf.evaluate(x=1.5, y=0) < 0

  # Rotate by 90 degrees
  cbf.rotate(sp.pi / 2)

  # After 90-degree rotation: the rectangle's width/height swap effectively
  # Point (0, 1.5) should now be inside
  val_inside = cbf.evaluate(x=0, y=1.5)
  assert val_inside < 0
  # Point (1.5, 0) should now be outside (was along width, now along height which is 2)
  val_outside = cbf.evaluate(x=1.5, y=0)
  assert val_outside > 0


def test_cbf_mul():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0, r=1)

  val_orig = cbf.evaluate(x=2, y=0)

  cbf_scaled = cbf * 3
  val_scaled = cbf_scaled.evaluate(x=2, y=0)
  assert abs(val_scaled - 3 * val_orig) < 1e-10

  # rmul
  cbf_scaled2 = 3 * cbf
  val_scaled2 = cbf_scaled2.evaluate(x=2, y=0)
  assert abs(val_scaled2 - 3 * val_orig) < 1e-10


def test_cbf_halfplane():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_halfplane().substitute(p_x=1, p_y=0, n_x=1, n_y=0)

  # Normal points in +x, plane at x=1
  # Points with x > 1 are safe (positive)
  assert cbf.evaluate(x=2, y=0) > 0
  # Points with x < 1 are unsafe (negative)
  assert cbf.evaluate(x=0, y=0) < 0
  # Boundary
  assert cbf.evaluate(x=1, y=0) == 0


def test_cbf_get_unique_symbol():
  factory = ControlBarrierFunctionFactory()
  s1 = factory.get_unique_symbol("test")
  s2 = factory.get_unique_symbol("test")
  assert s1 != s2
  assert s1.name == "test_1"
  assert s2.name == "test_2"


def test_cbf_get_unique_symbol_no_add():
  factory = ControlBarrierFunctionFactory()
  s1 = factory.get_unique_symbol("temp", add=False)
  s2 = factory.get_unique_symbol("temp", add=False)
  # Without adding, both should get the same name
  assert s1.name == s2.name == "temp_1"


def test_cbf_symbol_tracking():
  factory = ControlBarrierFunctionFactory()
  initial_count = len(factory.all_symbols)

  factory.disable_symbol_tracking()
  factory.get_unique_symbol("tracked")
  factory.enable_symbol_tracking()

  # Symbol added during disabled tracking should be reverted
  assert len(factory.all_symbols) == initial_count


def test_cbf_symbol_tracking_double_disable_raises():
  factory = ControlBarrierFunctionFactory()
  factory.disable_symbol_tracking()
  with pytest.raises(AssertionError):
    factory.disable_symbol_tracking()
  factory.enable_symbol_tracking()  # cleanup


def test_cbf_symbol_tracking_enable_without_disable_raises():
  factory = ControlBarrierFunctionFactory()
  with pytest.raises(AssertionError):
    factory.enable_symbol_tracking()


def test_cbf_evaluate_numpy_arrays():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0, r=1)

  x = np.array([0, 0, 2, 1])
  y = np.array([0, 0.5, 0, 0])
  vals = cbf.evaluate(x, y)

  assert vals.shape == (4,)
  assert vals[0] < 0  # inside
  assert vals[1] < 0  # inside
  assert vals[2] > 0  # outside
  assert vals[3] == 0  # boundary


def test_cbf_combine_invalid_method():
  factory = ControlBarrierFunctionFactory()
  cbf1 = factory.get_sphere()
  cbf2 = factory.get_sphere()
  with pytest.raises(ValueError, match="Invalid method"):
    factory.union([cbf1, cbf2], method="invalid")


def test_cbf_get_expr_substitute():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=1, c_y=2, r=3)

  expr_sub = cbf.get_expr(substitute=True)
  expr_nosub = cbf.get_expr(substitute=False)

  # Substituted expression should have fewer free symbols
  assert len(expr_sub.free_symbols) < len(expr_nosub.free_symbols)


def test_casadi_cbf_h():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0, r=1)
  ca_cbf = CasadiControlBarrierFunction(cbf)

  h_fn, h_symbols = ca_cbf.get_h()
  assert "x" in h_symbols
  assert "y" in h_symbols

  # Evaluate at boundary
  val = float(h_fn(x=1, y=0))
  assert abs(val) < 1e-10

  # Evaluate inside
  val = float(h_fn(x=0, y=0))
  assert val < 0


def test_casadi_cbf_grad_h():
  factory = ControlBarrierFunctionFactory()
  cbf = factory.get_sphere().substitute(c_x=0, c_y=0, r=1)
  ca_cbf = CasadiControlBarrierFunction(cbf)

  grad_fn, grad_symbols = ca_cbf.get_grad_h()
  assert "x" in grad_symbols
  assert "y" in grad_symbols

  # At point (1, 0), gradient should be [2, 0]
  grad = grad_fn(x=1, y=0)
  assert abs(float(grad[0]) - 2.0) < 1e-10
  assert abs(float(grad[1])) < 1e-10
