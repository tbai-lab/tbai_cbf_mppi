#!/usr/bin/env python3

import sympy as sp
import jax
import jax.numpy as jnp


def jit_expr(expr: sp.Expr, cse=True, symbols=None):
  """Take a sympy expression, convert it to a JAX-jitted callable.

  Works for both scalar and matrix expressions.
  """
  expr_symbols = [s.name for s in sorted(expr.free_symbols, key=lambda x: x.name)]
  symbols = symbols if symbols is not None else expr_symbols
  assert sorted(symbols) == expr_symbols, (
    f"Used symbols must be a permutation of the expression symbols: {symbols} != {expr_symbols}"
  )
  fn = sp.lambdify(symbols, expr, modules="jax", cse=cse)
  return jax.jit(fn)


def vectorize_expr(expr: sp.Expr, cse=True, symbols=None):
  """Take a sympy expression, convert it to a vmapped+jitted callable over arrays."""
  expr_symbols = [s.name for s in sorted(expr.free_symbols, key=lambda x: x.name)]
  symbols = symbols if symbols is not None else expr_symbols
  assert sorted(symbols) == expr_symbols, (
    f"Used symbols must be a permutation of the expression symbols: {symbols} != {expr_symbols}"
  )
  fn = sp.lambdify(symbols, expr, modules="jax", cse=cse)
  return jax.jit(jax.vmap(fn))
