import pytest
import numpy as np
import trapezion


def describe_trapz():

    @pytest.fixture
    def xs():
        return np.linspace(0, 1, 1000)

    @pytest.fixture
    def dxs():
        return np.gradient(np.linspace(0, 1, 1000))


    @pytest.fixture
    def func():
        return lambda x: np.array([x, x**2])[:, None, None] * np.array([1, 2, 3,])[None, :, None] * np.array([1, 1.01, 1.02, 1.03])[None, None, :]

    def its_fast(benchmark, func, xs, dxs):
        benchmark(trapezion.trapz, lambda x: func(x).flatten(), xs, dxs, (2, 3, 4,))


def describe_baseline():

    @pytest.fixture
    def trapz_ref():
        def _trapz_ref(function, xs, dxs):
            integral = 0
            for i, xi in enumerate(xs):
                if i == 0:
                    continue
                else:
                    integral += dxs[i] * (function(xs[i-1])+function(xi)) / 2
            return integral
        return _trapz_ref

    @pytest.fixture
    def xs():
        return np.linspace(0, 1, 1000)

    @pytest.fixture
    def dxs():
        return np.gradient(np.linspace(0, 1, 1000))


    @pytest.fixture
    def func():
        return lambda x: np.array([x, x**2])[:, None, None] * np.array([1, 2, 3,])[None, :, None] * np.array([1, 1.01, 1.02, 1.03])[None, None, :]

    def it_should_be_slower(benchmark, trapz_ref, func, xs, dxs):
        benchmark(trapz_ref, lambda x: func(x).flatten(), xs, dxs)
