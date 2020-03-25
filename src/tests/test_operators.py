import unittest
from src.lib.operators import Diff, D, PolyD, Poly, DataSplit, DataSplitOnIndex, Delay, MultipleDelay, \
    DataSplitIndexClip
from src.lib.variables import Variable, Domain, SymVariable, Field
import numpy as np


class TestDelayOperator(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 1, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x ** y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_Delay_order1(self):
        f_delay = Delay(delay=1, axis_name="x") * self.v
        assert np.allclose(f_delay.data[1:, :], self.v.data[:-1, :])
        assert f_delay.get_full_name() == "v(x - 1, y)".replace(' ', '')

        s_delay = Delay(delay=1, axis_name="x") * self.sym_v
        assert str(s_delay) == "v(x - 1, y)".replace(' ', '')

    def test_Delay_order2(self):
        order = 2
        f_delay = Delay(delay=order, axis_name="x") * self.v
        assert np.allclose(f_delay.data[order:, :], self.v.data[:-order, :])
        assert f_delay.get_full_name() == "v(x-{},y)".format(order)

        s_delay = Delay(delay=order, axis_name="x") * self.sym_v
        assert str(s_delay) == "v(x-{},y)".format(order)

    def test_Delay_combining_operations(self):
        order = 2
        f_delay = Delay(delay=1, axis_name="x") * Delay(delay=1, axis_name="x") * self.v
        assert np.allclose(f_delay.data[order:, :], self.v.data[:-order, :])
        assert f_delay.get_full_name() == "v(x-{},y)".format(order)

        s_delay = Delay(delay=1, axis_name="x") * Delay(delay=1, axis_name="x") * self.sym_v
        assert str(s_delay) == "v(x-{},y)".format(order)

    def test_Delay_combining_operations_2vars(self):
        order = 1
        f_delay = Delay(delay=1, axis_name="x") * Delay(delay=1, axis_name="y") * self.v
        assert np.allclose(f_delay.data[order:, order:], self.v.data[:-order, :-order])
        assert f_delay.get_full_name() == "v(x-{},y-{})".format(order, order)

        s_delay = Delay(delay=1, axis_name="x") * Delay(delay=1, axis_name="y") * self.sym_v
        assert str(s_delay) == "v(x-{},y-{})".format(order, order)

    if __name__ == '__main__':
        unittest.main()


class TestMultipleDelay(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 0.5, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x ** y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_over_variables(self):
        orders = [1, 2]
        multiple_d = MultipleDelay(delays_dict={'x': orders}) * self.v
        for var, order in zip(multiple_d.data, orders):
            assert np.allclose(var.data[order:, :], self.v.data[:-order, :])
            assert var.get_full_name() == "v(x-{},y)".format(order)

        # s_delay = Delay(delay=1, axis_name="x") * self.sym_v
        # assert str(s_delay) == "v(x-1,y)"

    def test_over_symvariables(self):
        orders = [1, 2]
        multiple_d = MultipleDelay(delays_dict={'x': orders}) * self.sym_v
        for svar, order in zip(multiple_d.data, orders):
            assert str(svar) == "v(x-{},y)".format(order)

    if __name__ == '__main__':
        unittest.main()


class TestDiffOperator(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 1, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x ** y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_Diff_order1(self):
        f_diff = Diff(difference_order=1, axis_name="x") * self.v
        assert all(f_diff.data[:, 0] == 0)
        assert all(f_diff.data[:, 1] == 2)
        assert f_diff.get_full_name() == "2*Derivative(v(x, y), x)".replace(' ', '')

        s_diff = Diff(difference_order=1, axis_name="x") * self.sym_v
        assert str(s_diff) == "-v(x - 1, y) + v(x + 1, y)".replace(' ', '')

    def test_Diff_order2(self):
        f_diff = Diff(difference_order=2, axis_name="x") * self.v
        assert all(f_diff.data[:, 0] == 0)
        assert all(f_diff.data[:, 1] == 0)
        assert all(f_diff.data[2:-2, 2] == 8)  # there is a difference in the borders for using np.gradient

        s_diff = Diff(difference_order=2, axis_name="x") * self.sym_v
        assert str(s_diff) == "-2*v(x, y) + v(x - 2, y) + v(x + 2, y)".replace(' ', '')

    def test_Diff_combining_operations(self):
        f_diff = Diff(difference_order=1, axis_name="x") * Diff(difference_order=1, axis_name="x") * self.v
        assert all(f_diff.data[:, 0] == 0)
        assert all(f_diff.data[:, 1] == 0)
        assert all(f_diff.data[2:-2, 2] == 8)

    # def test_Diff_over_lists(self):
    #     fg_diff = Diff(difference_order=1, axis_name="x") * self.FG
    #     assert all(fg_diff.F[0].f[:, 0] == 0)
    #     assert all(fg_diff.F[0].f[:, 1] == 1)
    #     assert all(fg_diff.F[1].f[:, 0] == 0)
    #     assert all(fg_diff.F[1].f[:, 1] == 1/2)

    # def test_Diff_Diff(self):
    #     fg_diff = (Diff(difference_order=1, axis_name="x") * self.f) / (Diff(difference_order=1, axis_name="x") * self.g)
    #     assert all(fg_diff.f[:, 1] == 2)


class TestDOperator(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 0.5, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x ** y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_Diff_order1(self):
        f_diff = D(derivative_order=1, axis_name="x") * self.v
        assert all(f_diff.data[:, 0] == 0)
        assert all(f_diff.data[:, 1] == 0.5)
        print(f_diff.name)

        s_diff = D(derivative_order=1, axis_name="x") * self.sym_v
        assert str(s_diff) == "-1.0*v(x - 1, y) + 1.0*v(x + 1, y)".replace(' ', '')

    def test_Diff_order2(self):
        f_diff = D(derivative_order=2, axis_name="x") * self.v
        assert all(f_diff.data[:, 0] == 0)
        assert all(f_diff.data[:, 1] == 0)
        assert all(f_diff.data[2:-2, 2] == 0.5)  # there is a difference in the borders for using np.gradient

        s_diff = D(derivative_order=2, axis_name="x") * self.sym_v
        assert str(s_diff) == "-2.0*v(x, y) + 1.0*v(x - 2, y) + 1.0*v(x + 2, y)".replace(' ', '')

    def test_Diff_combining_operations(self):
        f_diff = D(derivative_order=1, axis_name="x") * D(derivative_order=1, axis_name="x") * self.v
        assert all(f_diff.data[:, 0] == 0)
        assert all(f_diff.data[:, 1] == 0)
        assert all(f_diff.data[2:-2, 2] == 0.5)

        v_diff = D(derivative_order=1, axis_name="x") * D(derivative_order=1, axis_name="y") * self.v
        w_diff = D(derivative_order=1, axis_name="y") * D(derivative_order=1, axis_name="x") * self.v
        assert v_diff == w_diff

    if __name__ == '__main__':
        unittest.main()


class TestPolyD(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 0.5, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x ** y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_over_variables(self):
        polyv = PolyD(derivative_order_dict={"x": 2, "y": 2}) * self.v
        assert len(polyv) == 9

    def test_over_SymVariables(self):
        polyv = PolyD(derivative_order_dict={"x": 2, "y": 2}) * self.sym_v
        # [print(e) for e in polyv.data]
        assert len(polyv) == 9

    if __name__ == '__main__':
        unittest.main()


class TestPoly(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 0.5, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x ** y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_over_variables(self):
        polyv = Poly(polynomial_order=2) * Field([self.v, self.w, self.x])
        assert len(polyv) == 10
        equalto = {'x(x)**2', 'v(x, y)', 'v(x, y)**2', 'v(x, y)*x(x)', '1.00000000000000', 'x(x)'}
        equalto = {e.replace(' ', '') for e in equalto}
        assert set([e.get_full_name() for e in polyv.data]) == equalto

    def test_over_SymVariables(self):
        polyv = Poly(polynomial_order=2) * self.sym_v
        equalto = {"v(x, y)**2", "1", "v(x, y)"}
        equalto = {e.replace(' ', '') for e in equalto}
        assert set([str(e) for e in polyv.data]) == equalto
        assert len(polyv) == 3

    if __name__ == '__main__':
        unittest.main()


class TestDataSplit(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 0.5, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x ** y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_over_variables(self):
        datas = DataSplit(axis_percentage_dict={"x": 0.5, "y": 0.5}, axis_init_percentage_dict={"x": 0, "y": 0}) * \
                Field([self.v, self.w, self.x])
        assert len(datas) == 3
        assert datas.domain.shape["x"] == 5
        assert datas.domain.shape["y"] == 3

        datas = DataSplit(axis_percentage_dict={"x": 0.5, "y": 0.5}, axis_init_percentage_dict={"x": 0.2, "y": 0.5}) * \
                Field([self.v, self.w, self.x])
        assert len(datas) == 3
        assert datas.domain.shape["x"] == 5
        assert datas.domain.shape["y"] == 3
        assert datas.domain.lower_limits["x"] == 1
        assert datas.domain.lower_limits["y"] == 3

    def test_over_SymVariables(self):
        datas = DataSplit(axis_percentage_dict={"x": 0.5, "y": 0.5}, axis_init_percentage_dict={"x": 0, "y": 0}) * \
                self.sym_v
        assert len(datas) == 1
        assert datas.domain.shape["x"] == 5
        assert datas.domain.shape["y"] == 3

    if __name__ == '__main__':
        unittest.main()


class TestDataSplitOnIndex(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 0.5, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x ** y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_over_fields(self):
        datas = DataSplitOnIndex(axis_index_dict={"x": 3, "y": 3}) * Field([self.v, self.w, self.x])
        assert len(datas) == 3
        assert datas.domain.shape["x"] == 3
        assert datas.domain.shape["y"] == 3

    def test_over_SymFields(self):
        datas = DataSplitOnIndex(axis_index_dict={"x": 3, "y": 3}) * self.sym_v

        assert len(datas) == 1
        assert datas.domain.shape["x"] == 3
        assert datas.domain.shape["y"] == 3

    if __name__ == '__main__':
        unittest.main()


class TestDataSplitIndexClip(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 0.5, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x + y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_over_fields(self):
        datas = DataSplitIndexClip(axis_start_dict={"x": 3, "y": 3}, axis_end_dict={"x": 5, "y": 5}) * Field([self.v, self.w, self.x])
        assert len(datas) == 3
        assert datas.domain.shape["x"] == 2
        assert datas.domain.shape["y"] == 2

    def test_over_fields_len(self):
        datas = DataSplitIndexClip(axis_start_dict={"x": 3, "y": 3}, axis_len_dict={"x": 2, "y": 2}) * Field([self.v, self.w, self.x])
        assert len(datas) == 3
        assert datas.domain.shape["x"] == 2
        assert datas.domain.shape["y"] == 2

        datas = DataSplitIndexClip(axis_start_dict={"x": 1}, axis_len_dict={"x": 2}) * self.x
        assert len(datas) == 1
        assert datas.domain.shape["x"] == 2
        assert np.all(datas.data == np.array([0.5, 1]))

        datas = DataSplitIndexClip(axis_end_dict={"x": 3, "y": 3}, axis_len_dict={"x": 2, "y": 2}) * Field(
            [self.v, self.w, self.x])
        assert len(datas) == 3
        assert datas.domain.shape["x"] == 2
        assert datas.domain.shape["y"] == 2

        datas = DataSplitIndexClip(axis_end_dict={"x": -1}, axis_len_dict={"x": 2}) * self.x
        assert len(datas) == 1
        assert datas.domain.shape["x"] == 2
        assert np.all(datas.data == np.array([4, 4.5]))

        datas = DataSplitIndexClip(axis_end_dict={"x": 0}, axis_len_dict={"x": 2}) * self.x
        assert len(datas) == 1
        assert datas.domain.shape["x"] == 2
        assert np.all(datas.data == np.array([0, 0.5]))

        datas = DataSplitIndexClip(axis_start_dict={"x": 0}, axis_len_dict={"x": 2}) * self.x
        assert len(datas) == 1
        assert datas.domain.shape["x"] == 2
        assert np.all(datas.data == np.array([0, 0.5]))

    def test_over_SymFields(self):
        datas = DataSplitIndexClip(axis_start_dict={"x": 3, "y": 3}, axis_end_dict={"x": 5, "y": 5}) * self.sym_v

        assert len(datas) == 1
        assert datas.domain.shape["x"] == 2
        assert datas.domain.shape["y"] == 2

    if __name__ == '__main__':
        unittest.main()



