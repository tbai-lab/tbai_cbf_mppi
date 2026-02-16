#!/usr/bin/env python3

import jax.numpy as jnp
import numpy as np
import sympy as sp
import pytest
from tbai_cbf_mppi.jax.symperf_jax import jit_expr, vectorize_expr


def test_jit_expr_scalar():
  x, y = sp.symbols("x, y")
  expr = x**2 + y**2
  fn = jit_expr(expr)

  result = fn(2.0, 3.0)
  assert abs(float(result) - 13.0) < 1e-10


def test_jit_expr_returns_jnp():
  x, y = sp.symbols("x, y")
  expr = x + y
  fn = jit_expr(expr)

  result = fn(1.0, 2.0)
  assert isinstance(result, jnp.ndarray)


def test_jit_expr_custom_symbol_order():
  x, y = sp.symbols("x, y")
  expr = x - y
  fn = jit_expr(expr, symbols=["y", "x"])

  # With symbols=["y", "x"], first arg is y, second is x
  result = fn(1.0, 5.0)  # y=1, x=5 => 5 - 1 = 4
  assert abs(float(result) - 4.0) < 1e-10


def test_jit_expr_invalid_symbols():
  x, y = sp.symbols("x, y")
  expr = x + y
  with pytest.raises(AssertionError):
    jit_expr(expr, symbols=["x", "z"])


def test_jit_expr_matrix():
  x, y = sp.symbols("x, y")
  expr = sp.Matrix([x + y, x - y])
  fn = jit_expr(expr)

  result = fn(3.0, 1.0)
  assert abs(float(result[0, 0]) - 4.0) < 1e-10
  assert abs(float(result[1, 0]) - 2.0) < 1e-10


def test_jit_expr_trig():
  x = sp.Symbol("x")
  expr = sp.sin(x) ** 2 + sp.cos(x) ** 2
  fn = jit_expr(expr)

  for val in [0.0, 0.5, 1.0, 3.14]:
    assert abs(float(fn(val)) - 1.0) < 1e-6


def test_jit_expr_min():
  x, y = sp.symbols("x, y")
  expr = sp.Min(x, y)
  fn = jit_expr(expr)

  assert float(fn(2.0, 3.0)) == 2.0
  assert float(fn(5.0, 1.0)) == 1.0


def test_vectorize_expr_basic():
  x, y = sp.symbols("x, y")
  expr = x**2 + y**2
  fn = vectorize_expr(expr)

  x_arr = jnp.array([1.0, 2.0, 3.0])
  y_arr = jnp.array([0.0, 0.0, 0.0])
  result = fn(x_arr, y_arr)

  expected = jnp.array([1.0, 4.0, 9.0])
  assert jnp.allclose(result, expected)


def test_vectorize_expr_returns_jnp():
  x, y = sp.symbols("x, y")
  expr = x + y
  fn = vectorize_expr(expr)

  result = fn(jnp.array([1.0]), jnp.array([2.0]))
  assert isinstance(result, jnp.ndarray)


def test_vectorize_expr_custom_symbols():
  x, y = sp.symbols("x, y")
  expr = x - y
  fn = vectorize_expr(expr, symbols=["y", "x"])

  # With symbols=["y", "x"], first arg is y, second is x
  y_arr = jnp.array([1.0, 2.0])
  x_arr = jnp.array([5.0, 5.0])
  result = fn(y_arr, x_arr)  # x - y = [4, 3]
  assert jnp.allclose(result, jnp.array([4.0, 3.0]))


def test_vectorize_expr_invalid_symbols():
  x = sp.Symbol("x")
  expr = x**2

  with pytest.raises(AssertionError):
    vectorize_expr(expr, symbols=["z"])
