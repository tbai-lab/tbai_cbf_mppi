#!/usr/bin/env python3

import numpy as np
import sympy as sp
import pytest
from tbai_cbf_mppi.symperf import jit_expr, vectorize_expr


def test_jit_expr_scalar():
  x, y = sp.symbols("x, y")
  expr = x**2 + y**2
  fn = jit_expr(expr, backend="numpy")

  result = fn(2.0, 3.0)
  assert abs(result - 13.0) < 1e-10


def test_jit_expr_custom_symbol_order():
  x, y = sp.symbols("x, y")
  expr = x - y
  # Swap symbol order
  fn = jit_expr(expr, symbols=["y", "x"], backend="numpy")

  # With symbols=["y", "x"], first arg is y, second is x
  result = fn(1.0, 5.0)  # y=1, x=5 => 5 - 1 = 4
  assert abs(result - 4.0) < 1e-10


def test_jit_expr_invalid_symbols():
  x, y = sp.symbols("x, y")
  expr = x + y
  with pytest.raises(AssertionError):
    jit_expr(expr, symbols=["x", "z"], backend="numpy")


def test_jit_expr_invalid_backend():
  x = sp.Symbol("x")
  expr = x**2
  with pytest.raises(ValueError, match="Invalid backend"):
    jit_expr(expr, backend="invalid")


def test_jit_expr_matrix():
  x, y = sp.symbols("x, y")
  expr = sp.Matrix([x + y, x - y])
  fn = jit_expr(expr, backend="numpy")

  result = fn(3.0, 1.0)
  assert abs(result[0, 0] - 4.0) < 1e-10
  assert abs(result[1, 0] - 2.0) < 1e-10


def test_jit_expr_trig():
  x = sp.Symbol("x")
  expr = sp.sin(x) ** 2 + sp.cos(x) ** 2
  fn = jit_expr(expr, backend="numpy")

  # sin^2 + cos^2 = 1 for any x
  for val in [0.0, 0.5, 1.0, 3.14]:
    assert abs(fn(val) - 1.0) < 1e-10


def test_jit_expr_min():
  x, y = sp.symbols("x, y")
  expr = sp.Min(x, y)
  fn = jit_expr(expr, backend="numpy")

  assert fn(2.0, 3.0) == 2.0
  assert fn(5.0, 1.0) == 1.0


def test_vectorize_expr_basic():
  x, y = sp.symbols("x, y")
  expr = x**2 + y**2

  import numba
  fn = vectorize_expr(expr, nbtype=numba.float64)

  x_arr = np.array([1.0, 2.0, 3.0])
  y_arr = np.array([0.0, 0.0, 0.0])
  result = fn(x_arr, y_arr)

  expected = np.array([1.0, 4.0, 9.0])
  assert np.allclose(result, expected)


def test_vectorize_expr_custom_symbols():
  x, y = sp.symbols("x, y")
  expr = x - y

  import numba
  fn = vectorize_expr(expr, nbtype=numba.float64, symbols=["y", "x"])

  # With symbols=["y", "x"], first arg is y, second is x
  y_arr = np.array([1.0, 2.0])
  x_arr = np.array([5.0, 5.0])
  result = fn(y_arr, x_arr)  # x - y = [4, 3]
  assert np.allclose(result, [4.0, 3.0])


def test_vectorize_expr_invalid_symbols():
  x = sp.Symbol("x")
  expr = x**2

  import numba
  with pytest.raises(AssertionError):
    vectorize_expr(expr, nbtype=numba.float64, symbols=["z"])
