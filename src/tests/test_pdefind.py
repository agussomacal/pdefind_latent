import unittest
import sympy
from src.lib.operators import Diff, D, PolyD, Poly, DataSplit, DataSplitOnIndex,Identity
from src.lib.variables import Variable, Domain, SymVariable, Field
from src.lib.pdefind import PDEFinder, DataManager, StandardScalerForPDE
from src.lib.evaluators import evaluate_predictions

import numpy as np


class TestDataManager(unittest.TestCase):
    def setUp(self):
        sympy.init_printing(use_latex=True)
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

    def test_constructor(self):
        data_manager = DataManager()
        data_manager.add_variables(self.v)
        data_manager.add_variables([self.v, self.w])
        data_manager.add_field(Field(self.v))
        data_manager.add_regressors(self.x)
        assert len(data_manager.field) == 4
        assert len(data_manager.regressors) == 1
        data_manager.set_domain()
        assert len(data_manager.domain)

    def test_get_var(self):
        data_manager = DataManager()
        data_manager.add_variables([self.v])
        data_manager.add_regressors(self.x)
        data_manager.set_domain()
        data_manager.set_X_operator(lambda field: PolyD({"x": 1}) * field)  # (PolyD({"x": 1})
        data_manager.set_y_operator(lambda field: D(1, "x") * field)

        assert all(data_manager.get_X_dframe().columns == ['v(x,y)', 'x(x)'])
        assert all(data_manager.get_y_dframe().columns == ['1.0*Derivative(v(x,y),x)'])

    def test_get_sym(self):
        data_manager = DataManager()
        data_manager.add_variables([self.v])
        data_manager.add_regressors(self.x)
        data_manager.set_domain()

        data_manager.set_X_operator(lambda field: (PolyD({"x": 1}) * field))  # (PolyD({"x": 1})
        data_manager.set_y_operator(lambda field: D(2, "x") * field)
        assert str(data_manager.get_X_sym()) == "[v(x,y), -0.5*v(x-1,y)+0.5*v(x+1,y), x(x)]"
        assert str(data_manager.get_y_sym()) == "[-0.5*v(x,y)+0.25*v(x-2,y)+0.25*v(x+2,y)]"

        data_manager.set_X_operator(lambda field: (PolyD({"x": 1}) * field))  # (PolyD({"x": 1})
        data_manager.set_y_operator(lambda field: D(1, "x") * field)
        assert str(data_manager.get_X_sym()) == "[v(x,y), x(x)]"
        assert str(data_manager.get_y_sym()) == "[-0.5*v(x-1,y)+0.5*v(x+1,y)]"

    def test_getXy_eq(self):
        data_manager = DataManager()
        data_manager.add_variables([self.v])
        data_manager.add_regressors(self.x)
        data_manager.set_domain()
        data_manager.set_X_operator(lambda field: (PolyD({"x": 1}) * field))  # (PolyD({"x": 1})
        data_manager.set_y_operator(lambda field: D(2, "x") * field)

        # print(data_manager.get_Xy_eq()[0].data[1].sym_expression)
        assert str(data_manager.get_Xy_eq()[0]) == "[v(x,y), 1.0*Derivative(v(x,y),x), x(x)]"
        assert str(data_manager.get_Xy_eq()[1]) == "[1.0*Derivative(v(x,y),(x,2))]"


# class TestStandarScaler(unittest.TestCase):
#     def setUp(self):
#         self.domain = Domain(lower_limits_dict={"x": -20},
#                              upper_limits_dict={"x": 20},
#                              step_width_dict={"x": 0.0001})
#         self.v = Variable(self.domain.get_range("x")["x"],
#                           self.domain,
#                           domain2axis={"x": 0},
#                           variable_name="v")
#         self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))
#
#     def test_fit(self):
#         with_mean = True
#         with_std = True
#         X_scaler = StandardScalerForPDE(with_mean=with_mean, with_std=with_std)
#         y_scaler = StandardScalerForPDE(with_mean=with_mean, with_std=with_std)
#
#         data_manager = DataManager()
#         data_manager.add_variables(self.v)
#         data_manager.set_X_operator(lambda field: Poly(1) * field)  # (PolyD({"x": 1})
#         data_manager.set_y_operator(lambda field: D(2, "x") * field)
#
#         X_scaler.fit(data_manager.get_X_dframe())
#         y_scaler.fit(data_manager.get_y_dframe())
#
#         a = X_scaler.sym_var_transform(data_manager.get_X_sym()).dot(np.ones(len(data_manager.get_X_sym())))
#         assert a.evaluate({"x": 1}) == np.sum(X_scaler.transform(data_manager.get_X_dframe().values), axis=1)[1]
#         b = y_scaler.sym_var_inverse_transform(a)
#         assert b.evaluate({"x": 1})[0] - y_scaler.inverse_transform(np.sum(X_scaler.transform(data_manager.get_X_dframe().values), axis=1))[1] < 1e-10


class TestPDEFinder(unittest.TestCase):
    def setUp(self):
        self.domain = Domain(lower_limits_dict={"x": -10},
                             upper_limits_dict={"x": 10},
                             step_width_dict={"x": 0.001})
        self.v = Variable(np.sin(self.domain.get_range("x")["x"]),
                          self.domain,
                          domain2axis={"x": 0},
                          variable_name="v")
        self.x = Variable(self.domain.get_range("x")["x"],
                          self.domain,
                          domain2axis={"x": 0},
                          variable_name="x")
        self.sym_v = SymVariable(*SymVariable.get_init_info_from_variable(self.v))

    def test_fit(self):
        data_manager = DataManager()
        data_manager.add_variables(self.v)
        data_manager.set_X_operator(lambda field: Poly(3) * (PolyD({"x": 1}) * field))  # (PolyD({"x": 1})
        data_manager.set_y_operator(lambda field: D(2, "x") * field)

        pde_finder = PDEFinder(with_mean=True, with_std=True)
        pde_finder.set_fitting_parameters(cv=20, n_alphas=100, alphas=None)
        pde_finder.fit(data_manager.get_X_dframe(), data_manager.get_y_dframe())
        print(pde_finder.coefs_)  # strange th value obtained

        print((pde_finder.transform(data_manager.get_X_dframe()) - data_manager.get_y_dframe()).abs().mean().values)
        assert np.max((pde_finder.transform(data_manager.get_X_dframe()) - data_manager.get_y_dframe()).abs().mean().values) < 1e-5

        res = pde_finder.get_equation(*data_manager.get_Xy_eq())
        print(res)

        res = pde_finder.get_equation(data_manager.get_X_sym(), data_manager.get_y_sym())
        print(res)

    def test_fit_2(self):
        data_manager = DataManager()
        data_manager.add_variables(self.v)
        data_manager.add_variables(self.v**2)
        data_manager.set_X_operator(lambda field: Poly(3) * (PolyD({"x": 1}) * field))  # (PolyD({"x": 1})
        data_manager.set_y_operator(lambda field: D(2, "x") * field)

        pde_finder = PDEFinder(with_mean=True, with_std=True)
        pde_finder.set_fitting_parameters(cv=10, n_alphas=100, alphas=None)
        pde_finder.fit(data_manager.get_X_dframe(), data_manager.get_y_dframe())
        print(pde_finder.coefs_)  # strange th value obtained

        print((pde_finder.transform(data_manager.get_X_dframe()) - data_manager.get_y_dframe()).abs().mean().values)
        assert np.max((pde_finder.transform(data_manager.get_X_dframe()) - data_manager.get_y_dframe()).abs().mean().values) < 1e-5

        res = pde_finder.get_equation(*data_manager.get_Xy_eq())
        print(res)

        res = pde_finder.get_equation(data_manager.get_X_sym(), data_manager.get_y_sym())
        print(res)

        # import matplotlib.pylab as plt
        # plt.plot(pde_finder.transform(data_manager.get_X_dframe(testSplit)))
        # plt.plot(data_manager.get_y_dframe(testSplit))
        # plt.show()
        # print("d")

        # import matplotlib.pylab as plt
        # plt.plot(pde_finder.transform(data_manager.get_X_dframe(testSplit)))
        # plt.plot(data_manager.get_y_dframe(testSplit))
        # plt.show()
        # print("d")

    def test_predict_field_2(self):
        trainSplit = DataSplit({"x": 0.7})
        testSplit = DataSplit({"x": 0.3}, {"x": 0.7})

        data_manager = DataManager()
        data_manager.add_variables(self.v)
        data_manager.add_variables(self.x)
        data_manager.set_X_operator(lambda field: PolyD({"x": 1}) * field)  # (PolyD({"x": 1})
        data_manager.set_y_operator(lambda field: D(2, "x") * field)

        pde_finder = PDEFinder(with_mean=True, with_std=True)
        pde_finder.set_fitting_parameters(cv=20, n_alphas=100, alphas=None)
        pde_finder.fit(data_manager.get_X_dframe(trainSplit), data_manager.get_y_dframe(trainSplit))
        print(pde_finder.coefs_)  # strange th value obtained

        # warning!!!
        predictions_df = pde_finder.predict([DataSplitOnIndex({"x": 5}) * testSplit,
                                             DataSplitOnIndex({"x": 20}) * testSplit],
                                            data_manager,
                                            starting_point={"x": -1},
                                            domain_variable2predict="x",
                                            horizon=10)

        print(predictions_df)

    def test_integrate(self):
        trainSplit = DataSplit({"x": 0.7})
        testSplit = DataSplit({"x": 0.3}, {"x": 0.7})

        data_manager = DataManager()
        data_manager.add_variables(self.v)
        data_manager.set_X_operator(lambda field: PolyD({"x": 1}) * field)  # (PolyD({"x": 1})
        data_manager.set_y_operator(lambda field: D(2, "x") * field)
        data_manager.set_domain()

        pde_finder = PDEFinder(with_mean=True, with_std=True)
        pde_finder.set_fitting_parameters(cv=20, n_alphas=100, alphas=None)
        pde_finder.fit(data_manager.get_X_dframe(trainSplit), data_manager.get_y_dframe(trainSplit))
        print(pde_finder.coefs_)  # strange th value obtained

        # warning!!!
        predictions_df = pde_finder.integrate([DataSplitOnIndex({"x": 5}) * testSplit,
                                               DataSplitOnIndex({"x": 20}) * testSplit],
                                              data_manager,
                                              starting_point={"x": -1},
                                              domain_variable2predict="x",
                                              horizon=10)

        print(predictions_df)

    def test_evaluator(self):
        trainSplit = DataSplit({"x": 0.7})
        testSplit = DataSplit({"x": 0.3}, {"x": 0.7})

        data_manager = DataManager()
        data_manager.add_variables(self.v)
        data_manager.add_variables(self.x)
        data_manager.set_X_operator(lambda field: PolyD({"x": 1}) * field)  # (PolyD({"x": 1})
        data_manager.set_y_operator(lambda field: D(3, "x") * field)

        pde_finder = PDEFinder(with_mean=True, with_std=True)
        pde_finder.set_fitting_parameters(cv=20, n_alphas=100, alphas=None)
        pde_finder.fit(data_manager.get_X_dframe(trainSplit), data_manager.get_y_dframe(trainSplit))
        print(pde_finder.coefs_)  # strange th value obtained

        real, pred = evaluate_predictions(pde_finder,
                                          data_split_operator=testSplit,
                                          dm= data_manager,
                                          starting_point={"x": -1},
                                          domain_variable2predict="x",
                                          horizon=10,
                                          num_evaluations=1)

        assert np.mean(real.drop(["random_split", "method"], axis=1).values - pred.drop(["method"], axis=1).values[1:, :]) < 0.001
