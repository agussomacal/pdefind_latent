import unittest
from src.lib.variables import Domain, Variable, SymVariable, Field, SymField
import numpy as np
import sympy
import copy
from collections import OrderedDict


class TestDomain(unittest.TestCase):
    def setUp(self):  # -> None:
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 1, "y": 1})

        self.domain2 = Domain(lower_limits_dict={"x": 1, "y": 0},
                              upper_limits_dict={"x": 4, "y": 7},
                              step_width_dict={"x": 1, "y": 1})

    def testShape(self):
        assert self.domain.get_shape() == {'y': 6.0, 'x': 5.0}

    def testRange(self):
        assert np.all(self.domain.get_range("x")["x"] == np.arange(0, 5))

    def testSubdomain(self):
        assert self.domain.get_subdomain("x").get_shape("x") == {"x": 5}
        assert self.domain.get_subdomain({"x": [0, 2]}).get_shape("x") == {"x": 2}

    def testis_index_in_range(self):
        assert self.domain.is_index_in_range({"x": 1})
        assert not self.domain.is_index_in_range({"x": 10})

    def test_mul(self):
        assert (self.domain * self.domain2).get_shape() == {'y': 6.0, 'x': 3.0}
        assert (self.domain * self.domain2.get_subdomain({"x": [0, 2]})).get_shape() == {'x': 1.0, 'y': 6.0}

    if __name__ == '__main__':
        unittest.main()


class TestVariables(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 1, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x**y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="w")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")

    def test_Variable(self):
        assert self.v.domain.shape["x"] == self.domain.shape["x"]
        assert self.v.domain.shape["y"] == self.domain.shape["y"]

    def test_Variable_mul_1(self):
        a = self.x * self.y
        a.reorder_axis(self.domain)
        a.reorder_axis({"x": 0, "y": 1})
        mg = [[dx*dy for dy in self.y.data] for dx in self.x.data]
        assert np.all(a.data == mg)

    def test_Variable_mul_2(self):
        a = (self.v * self.x)
        a.reorder_axis(self.domain)
        b = (self.x * self.v)
        b.reorder_axis(self.domain)
        assert np.all(a.data == b.data)

    def test_add(self):
        a = (self.v + self.w)
        a.reorder_axis(self.domain)
        self.v.reorder_axis(self.domain)
        self.w.reorder_axis(self.domain)
        assert np.all(a.data == (self.v.data + self.w.data))
        assert np.all((self.v + 2).data == self.v.data + 2)

    def test_sub(self):
        a = (self.v - self.w)
        a.reorder_axis(self.domain)
        assert np.all(a.data == 0)
        assert np.all((self.v - 2).data == self.v.data - 2)

    def test_eval(self):
        assert self.v.eval({"x": 0, "y": 0}) == 1

    def test_index_eval(self):
        assert self.v.index_eval({"x": 0, "y": 0}) == 1
        assert self.v.index_eval({"x": -1, "y": -1}) == 4**5

    def test_pow(self):
        a = self.v ** self.w
        a.reorder_axis(self.domain)
        self.v.reorder_axis(self.domain)
        self.w.reorder_axis(self.domain)
        assert np.all(a.data == (self.v.data ** self.w.data))

        a = self.v ** 2
        a.reorder_axis(self.domain)
        self.v.reorder_axis(self.domain)
        assert np.all(a.data == (self.v.data ** 2))

        a = 2 ** self.v
        a.reorder_axis(self.domain)
        self.v.reorder_axis(self.domain)
        assert np.all(a.data == (2 ** self.v.data))

    def test_get_subset_from_index_range(self):
        new_var = self.v.get_subset_from_index_limits({"x": [0, 2], "y": [0, 2]})
        assert new_var.shape == (2, 2)

    if __name__ == '__main__':
        unittest.main()


class TestSymVariables(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 1, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x**y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="w")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_w = SymVariable(*SymVariable.get_init_info_from_variable(self.w))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_init(self):
        assert str(self.sym_v) == "v(x, y)".replace(' ', '')
        assert self.sym_v.evaluate({"x": 0, "y": 0}) == 1

    def test_shift(self):
        sym_v2 = copy.deepcopy(self.sym_v)
        sym_v2 = SymVariable.shift(sym_v2, {"x": -1, "y": 2})

        assert str(sym_v2) == "v(x - 1, y + 2)".replace(' ', '')
        assert sym_v2.evaluate({"x": 3, "y": 0}) == 4
        assert str(self.sym_v) == "v(x, y)".replace(' ', '')
        assert self.sym_v.evaluate({"x": 3, "y": 0}) == 1

    def test_add(self):
        sym_v2 = copy.deepcopy(self.sym_v)
        sym_v2 = SymVariable.shift(sym_v2, {"x": -1, "y": 2})
        sym_new = sym_v2 + self.sym_v

        assert str(sym_new) == "v(x, y) + v(x - 1, y + 2)".replace(' ', '')
        assert sym_new.evaluate({"x": 3, "y": 0}) == 5

        sym_new = self.sym_v - self.sym_x
        assert str(sym_new) == "v(x, y) - x(x)".replace(' ', '')

    def test_sub(self):
        a = (self.sym_v - self.sym_w)
        assert str(a) == "v(x, y) - w(x, y)".replace(' ', '')
        assert str(self.sym_v - 2) == "v(x, y) - 2".replace(' ', '')

    def test_mul(self):
        sym_v2 = copy.deepcopy(self.sym_v)
        sym_v2 = SymVariable.shift(sym_v2, {"x": -1, "y": 2})
        sym_new = sym_v2 * self.sym_v

        assert str(sym_new) == "v(x, y)*v(x - 1, y + 2)".replace(' ', '')
        assert sym_new.evaluate({"x": 3, "y": 0}) == 4

        sym_new = self.sym_v * self.sym_x
        assert str(sym_new) == "v(x, y)*x(x)".replace(' ', '')

    def test_simplify(self):
        b = copy.deepcopy(self.sym_v)
        b.simplify()
        assert b == self.sym_v

    def test_pow(self):
        a = self.sym_v ** self.sym_w
        assert str(a) == "v(x, y)**w(x, y)".replace(' ', '')

        a = self.sym_v ** 2
        assert str(a) == "v(x, y)**2".replace(' ', '')

        a = 2 ** self.sym_v
        assert str(a) == "2**v(x, y)".replace(' ', '')

    def test_evaluate(self):
        a = copy.deepcopy(self.sym_v)
        assert a.evaluate({"x": 4, "y": 5}) == 4**5

        assert str(a.evaluate({"x": 5, "y": 5})) == "v(5, 5)"
        assert len(sympy.solve(a.evaluate({"x": 5, "y": 5})**2-10)) == 2

    if __name__ == '__main__':
        unittest.main()


class TestField(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 1, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x**y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="w")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")

    def test_constructor(self):
        fv = Field()
        assert len(fv) == 0
        fv = Field(self.v)
        assert len(fv) == 1
        fv.append(Field(self.w))
        assert len(fv) == 2

    def test_topandas(self):
        fv = Field(self.v)
        fv.append(Field(self.w))
        print(fv.to_pandas().shape)
        assert fv.to_pandas().shape == (self.domain.get_shape("y")["y"] * self.domain.get_shape("x")["x"], 2)

    def test_get_subset_from_index_limits(self):
        fv = Field(self.v)
        new_fv = fv.get_subset_from_index_limits({"x": [0, 2], "y": [0, 2]})
        assert new_fv.domain.get_shape() == {"x": 2, "y": 2}
        assert fv.domain.get_shape() != {"x": 2, "y": 2}

    def test_str(self):
        fvw = Field([self.v, self.w])
        fxy = Field([self.x, self.y])

        assert str(fvw) == "[v(x,y), w(x,y)]"
        assert str(fxy) == "[x(x), y(y)]"

    def test_mul(self):
        fvw = Field([self.v, self.w])
        fxy = Field([self.x, self.y])

        assert len(fvw * fxy) == 2
        assert isinstance(fvw * fxy, Field)

    def test_add(self):
        fvw = Field([self.v, self.w])
        fxy = Field([self.x, self.y])

        assert len(fvw + fxy) == 2
        assert isinstance(fvw + fxy, Field)

    def test_subs(self):
        fvw = Field([self.v, self.w])
        fxy = Field([self.x, self.y])

        assert len(fvw - fxy) == 2
        assert isinstance(fvw - fxy, Field)
        assert isinstance(fvw - 2, Field)

    def test_dot(self):
        fvw = Field([self.v, self.w])
        fxy = Field([self.x, self.y])

        assert len(fvw.dot(fxy)) == 1
        assert isinstance(fvw.dot(fxy), Variable)

    def test_pow(self):
        fvw = Field([self.v, self.w])

        assert len(fvw ** fvw) == 2
        assert len(fvw ** 2) == 2
        assert len(2 ** fvw) == 2

    if __name__ == '__main__':
        unittest.main()


class TestSymField(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": 0, "y": 0},
                             upper_limits_dict={"x": 5, "y": 6},
                             step_width_dict={"x": 1, "y": 1})

        self.f = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        self.g = np.zeros((self.domain.get_shape("x")["x"], self.domain.get_shape("y")["y"]))
        for i, x in enumerate(self.domain.get_range("x")["x"]):
            for j, y in enumerate(self.domain.get_range("y")["y"]):
                self.f[i, j] = x**y
                self.g[i, j] = self.f[i, j]

        self.v = Variable(self.f, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="v")
        self.w = Variable(self.g, self.domain, domain2axis={"x": 0, "y": 1}, variable_name="w")

        self.x = Variable(self.domain.get_range("x")["x"], self.domain.get_subdomain("x"),
                          domain2axis={"x": 0}, variable_name="x")
        self.y = Variable(self.domain.get_range("y")["y"], self.domain.get_subdomain("y"),
                          domain2axis={"y": 0}, variable_name="y")

        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
        self.sym_x = SymVariable(*SymVariable.get_init_info_from_variable(self.x))

    def test_constructor(self):
        fv = SymField()
        assert len(fv) == 0
        fv = SymField(self.sym_v)
        assert len(fv) == 1
        fv.append(SymField(self.sym_x))
        assert len(fv) == 2
        fv.append(self.sym_x)
        assert len(fv) == 3

    def test_str(self):
        fvw = SymField([self.v, self.w])
        fxy = SymField([self.x, self.y])

        assert str(fvw) == "[v(x,y), w(x,y)]"
        assert str(fxy) == "[x(x), y(y)]"

    def test_mul(self):
        fvw = SymField([self.v, self.w])
        fxy = SymField([self.x, self.y])

        assert len(fvw * fxy) == 2
        assert isinstance(fvw * fxy, SymField)

    def test_add(self):
        fvw = SymField([self.v, self.w])
        fxy = SymField([self.x, self.y])

        assert len(fvw + fxy) == 2
        assert isinstance(fvw + fxy, SymField)

    def test_subs(self):
        fvw = SymField([self.v, self.w])
        fxy = SymField([self.x, self.y])

        assert len(fvw - fxy) == 2
        assert isinstance(fvw - fxy, SymField)
        assert isinstance(fvw - 2, SymField)

        # [print(v) for v in (fvw - 2).data]
        # [print(v) for v in fvw.data]

    def test_pow(self):
        fvw = Field([self.v, self.w])

        assert len(fvw ** fvw) == 2
        assert len(fvw ** 2) == 2
        assert len(2 ** fvw) == 2

    def test_dot(self):
        fvw = SymField([self.v, self.w])
        fxy = SymField([self.x, self.y])

        assert len(fvw.dot(fxy)) == 1
        assert isinstance(fvw.dot(fxy), SymVariable)
        assert str(fvw.dot(fxy)) == "v(x, y)*x(x) + w(x, y)*y(y)".replace(' ', '')

    def test_matmul(self):
        fvw = SymField([self.v, self.w])
        a = fvw.matmul(np.array([[0, 1], [0, 1]]))
        assert len(a) == 2
        assert isinstance(a, SymField)
        assert str(a) == '[0, 1.0*v(x,y)+1.0*w(x,y)]'

    def test_evaluate(self):
        fvw = SymField([self.v, self.w])
        assert all(fvw.evaluate({"x": 0, "y": 0}) == np.ones(2))

    def test_simplify(self):
        fvw = SymField([self.v, self.w])
        print(fvw)
        fvw.simplify()
        print(fvw)

    if __name__ == '__main__':
        unittest.main()
