import unittest
from src.lib.algorithms import polynomial_order_cartprod, get_lag_from_sym_expression, get_func_for_ode, \
    get_func_for_pde
import sympy
import numpy as np
from collections import OrderedDict

from variables import Variable, Domain


class TestAlgorithms(unittest.TestCase):
    def test_polynomial_order_cartprod(self):
        cart_prod_dict = polynomial_order_cartprod(OrderedDict([("x", 3), ("y", 2)]))
        for e in cart_prod_dict.items():
            print(e)

        assert set(cart_prod_dict[(0, 1)].keys()) == {(1, 1), (0, 2)}
        assert set(cart_prod_dict[(1, 1)].keys()) == {(2, 1), (1, 2)}
        assert set(cart_prod_dict[(0, 0)].keys()) == {(1, 0), (0, 1)}
        assert set(cart_prod_dict[(3, 2)].keys()) == set()

    def test_get_lag_from_sym_expression(self):
        x = sympy.Symbol("x")
        f = sympy.Function("f")(x)
        assert get_lag_from_sym_expression(f**2+f+f.subs(x, x+2)+f.subs(x,x-1)) == ({'x': -1}, {'x': 2})

    def test_get_func_for_ode(self):
        x = sympy.Symbol("x")
        f = sympy.Function("f")(x)
        g = sympy.Function("g")(x)

        sym_x_expression = f**2+f-x
        func = get_func_for_ode(sym_x_expression, f.diff("x", 1))
        assert func([1], 1) == [1]

        sym_x_expression = f**2+f.diff()*f.diff("x", 3)-x+f.diff("x", 2)
        func = get_func_for_ode(sym_x_expression, f.diff("x", 4))
        assert func([1, 1, 1, 1], 1) == [1, 1, 1, 2]

        sym_x_expression = f**2+f.diff()*f.diff("x", 3)-x
        func = get_func_for_ode(sym_x_expression, f.diff("x", 4))
        assert func([1, 1, 1, 1], 1) == [1, 1, 1, 1]

        # sym_x_expression = g**2+f.diff()*f.diff("x", 3)-x
        # sym_x_expression = g**2+f.diff()*f.diff("x", 3)-x
        # func = get_func_for_ode(sym_x_expression, f.diff("x", 4))
        # print(func([1, 1, 1, 1], [1]))

    def test_get_func_for_pde(self):
        integration_steps = 10

        t = sympy.Symbol("t")
        x = sympy.Symbol("x")
        var_name = x.name
        f = sympy.Function("f")(x)
        g = sympy.Function("g")(t, x)

        g.subs({x: x + 2, t: t-1}).atoms()

        sym_x_expression = x
        func = get_func_for_pde(sym_x_expression, f.diff(x, 1))
        dom = Domain(lower_limits_dict={var_name: 1}, upper_limits_dict={var_name: 1}, step_width_dict={var_name: 0.1})
        var = Variable(data=[1], domain=dom, domain2axis={var_name: 0}, variable_name='f')
        res_var = func(data0=var, boundary=[], integration_steps=integration_steps)
        assert res_var.data == dom.step_width[var_name]*np.arange(integration_steps) + dom.lower_limits[var_name]


    if __name__ == '__main__':
        unittest.main()