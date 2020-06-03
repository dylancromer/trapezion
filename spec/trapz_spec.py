import pytest
import numpy as np
import trapezion


def describe_trapz():

    def it_can_integrate_a_simple_function():
        xs = np.linspace(0, 1, 100)
        dxs = np.gradient(xs)
        assert np.allclose(
            trapezion.trapz(lambda x: np.array([x**2]), xs, dxs, (1,)),
            1/3,
            rtol=1e-2,
        )

    def it_can_integrate_a_many_dimensional_function():
        def test_func(x):
            return (np.array([x, x**2])[:, None] * np.array([1, 2, 3])[None, :])


        xs = np.linspace(0, 1, 100)
        dxs = np.gradient(xs)

        result = trapezion.trapz(lambda x: test_func(x).flatten(), xs, dxs, (2, 3))
        assert np.allclose(result, np.array([[1/2, 1, 3/2], [1/3, 2/3, 1]]), rtol=1e-2)
