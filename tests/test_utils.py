import time
import numpy as np
import pytest
from tbai_cbf_mppi.utils import remove_kwargs, remove_outliers, PerformanceTimer, get_ellipse_points, Rate


def test_remove_kwargs():
  def example_func(a, b, d=4, c=3):
    return a * b / c**d

  def example_func_2(a, b, c=3, d=4):
    return example_func(a, d=d, c=c, b=b)

  fn, old_source, new_source = remove_kwargs(example_func_2, locals=locals(), globals=globals())
  assert new_source == "def example_func_2(a, b, c=3, d=4):\n    return example_func(a, b, d, c)"

  assert fn(1, 2, 3, 4) == 1 * 2 / 3**4
  assert fn(1, 2, d=4, c=3) == example_func(1, 2, 4, 3)


def test_remove_outliers_basic():
  data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
  result = remove_outliers(data, lower_percentile=5, upper_percentile=90)
  assert 100 not in result
  assert all(x in data for x in result)


def test_remove_outliers_no_removal():
  data = [5, 5, 5, 5, 5]
  result = remove_outliers(data, lower_percentile=0, upper_percentile=100)
  # All values are the same, percentile bounds equal the value,
  # strict inequality means all are removed
  assert len(result) == 0


def test_remove_outliers_invalid_percentiles():
  with pytest.raises(AssertionError):
    remove_outliers([1, 2, 3], lower_percentile=-1)
  with pytest.raises(AssertionError):
    remove_outliers([1, 2, 3], upper_percentile=101)
  with pytest.raises(AssertionError):
    remove_outliers([1, 2, 3], lower_percentile=90, upper_percentile=10)


def test_performance_timer_basic():
  timer = PerformanceTimer(unit="s")
  timer.tick()
  time.sleep(0.01)
  timer.tock()

  assert len(timer.get_data()) == 1
  assert timer.mean > 0
  assert timer.min > 0
  assert timer.max > 0
  assert timer.median > 0
  assert timer.std >= 0


def test_performance_timer_units():
  timer_ms = PerformanceTimer(unit="ms")
  timer_ms.tick()
  time.sleep(0.01)
  timer_ms.tock()

  timer_s = PerformanceTimer(unit="s")
  timer_s.tick()
  time.sleep(0.01)
  timer_s.tock()

  # ms value should be ~1000x the s value
  assert timer_ms.mean > timer_s.mean * 100


def test_performance_timer_invalid_unit():
  with pytest.raises(AssertionError):
    PerformanceTimer(unit="hours")


def test_performance_timer_multiple_samples():
  timer = PerformanceTimer(unit="us")
  for _ in range(5):
    timer.tick()
    timer.tock()

  assert len(timer.get_data()) == 5
  assert timer.min <= timer.mean <= timer.max


def test_performance_timer_str():
  timer = PerformanceTimer(unit="ms")
  assert "ms" in str(timer)
  assert "0" in str(timer)  # num_samples=0


def test_performance_timer_remove_outliers():
  timer = PerformanceTimer(unit="s")
  timer.execution_times = [0.01, 0.01, 0.01, 0.01, 100.0]
  timer.remove_outliers_(lower_percentile=5, upper_percentile=90)
  assert 100.0 not in timer.get_data()


def test_get_ellipse_points_shape():
  mean = np.array([0, 0])
  cov = np.eye(2)
  x, y = get_ellipse_points(mean, cov, num_points=50)
  assert x.shape == (50,)
  assert y.shape == (50,)


def test_get_ellipse_points_centered():
  mean = np.array([0, 0])
  cov = np.eye(2)
  x, y = get_ellipse_points(mean, cov, num_points=100)
  # Points should be roughly centered at mean
  assert abs(np.mean(x)) < 0.1
  assert abs(np.mean(y)) < 0.1


def test_get_ellipse_points_scaled():
  mean = np.array([0, 0])
  cov_small = np.eye(2) * 0.01
  cov_large = np.eye(2) * 100

  x_small, y_small = get_ellipse_points(mean, cov_small)
  x_large, y_large = get_ellipse_points(mean, cov_large)

  # Larger covariance should produce larger ellipse
  assert np.max(np.abs(x_large)) > np.max(np.abs(x_small))
  assert np.max(np.abs(y_large)) > np.max(np.abs(y_small))


def test_rate_sleep():
  rate = Rate(10)  # 10 Hz = 0.1s period
  start = time.perf_counter()
  rate.sleep()
  elapsed = time.perf_counter() - start
  assert 0.08 < elapsed < 0.15


def test_rate_accounts_for_elapsed():
  rate = Rate(10)  # 10 Hz
  time.sleep(0.05)  # simulate work taking 50ms
  start = time.perf_counter()
  rate.sleep()
  elapsed = time.perf_counter() - start
  # Should sleep about 50ms less than the full 100ms period
  assert elapsed < 0.08
