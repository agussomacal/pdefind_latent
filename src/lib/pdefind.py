import copy
import itertools
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import sympy
from scipy.integrate import odeint
from sklearn.linear_model import LassoCV, Lasso, LinearRegression, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import joblib
import odespy
from tqdm import tqdm

from src.lib.evaluators import rsquare, error
from src.lib.algorithms import get_lag_from_sym_expression, get_func_for_ode, get_sorted_derivative_atoms, \
    get_sorted_derivative_atoms2
from src.lib.operators import Identity, PolyD, DataSplitOnIndex, DataSplitIndexClip
from src.lib.variables import Variable, Domain, SymVariable, Field, SymField


##########################################################
#                       DataManager
##########################################################
class DataManager:
    def __init__(self):
        self.X_operator = lambda Field, regressors: Field
        self.y_operator = lambda Field: Field

        self.domain = Domain()

        self.field = Field()
        self.regressors = Field()
        self.sym_regressors = SymField()
        self.sym_field = SymField()

    def set_domain(self, domain_info=None):
        if domain_info is None:
            # in case none is passed then it will get the maximum domain from the variables of the field.
            self.field.set_domain()
            self.domain = self.field.domain
        elif isinstance(domain_info, dict):
            if ("lower_limits_dict" in domain_info.keys()) and ("upper_limits_dict" in domain_info.keys()) and \
                    ("step_width_dict" in domain_info.keys()):
                lower_limits_dict = domain_info["lower_limits_dict"]
                upper_limits_dict = domain_info["upper_limits_dict"]
                step_width_dict = domain_info["step_width_dict"]
            elif all([isinstance(element, np.ndarray) for element in domain_info.values()]):
                lower_limits_dict = {};
                upper_limits_dict = {};
                step_width_dict = {}
                for axis_name, range_vals in domain_info.items():
                    lower_limits_dict[axis_name] = range_vals[0]
                    upper_limits_dict[axis_name] = range_vals[-1]
                    step_width_dict[axis_name] = range_vals[1] - range_vals[0]
            else:
                Exception("imput should be or dict of ranges or dict of limits and steps")
            self.domain = Domain(lower_limits_dict, upper_limits_dict, step_width_dict)
        elif isinstance(domain_info, Domain):
            self.domain = copy.deepcopy(domain_info)

    def add_variables(self, variables):  # type:((list| Variable)) -> None
        self.field.append(variables)
        self.sym_field.append(variables)

    def add_field(self, field):
        self.field.append(field)
        self.sym_field.append(field)

    def add_regressors(self, regressors):  # : (Variable, Field)
        self.regressors.append(regressors)
        self.sym_regressors.append(regressors)

    def set_X_operator(self, operator):
        self.X_operator = operator

    def set_y_operator(self, operator):
        self.y_operator = operator

    @staticmethod
    def filter_yvars_in_Xvars(y_field, x_field):  # type: (Field, Field) -> Field
        y_var_names = [var.get_full_name() for var in y_field.data]
        return Field([var for var in x_field.data if var.get_full_name() not in y_var_names])

    @staticmethod
    def filter_ysymvars_in_Xsymvars(y_field, x_field):  # type: (SymField, SymField) -> SymField
        y_sym_expressions = [str(sym_var) for sym_var in y_field.data]
        return SymField([symvar for symvar in x_field.data if str(symvar) not in y_sym_expressions])

    def get_X_sym(self, split_operator=Identity()):
        """
        gets the simbolic expression of X
        :return:
        """
        sym_X = self.X_operator(split_operator * self.sym_field, split_operator * self.sym_regressors if self.sym_regressors != [] else self.sym_regressors)

        return self.filter_ysymvars_in_Xsymvars(self.get_y_sym(split_operator), x_field=sym_X)

    def get_y_sym(self, split_operator=Identity()):
        """
        gets the simbolic expression of y
        :return:
        """
        return self.y_operator(split_operator * self.sym_field)

    def get_X(self, split_operator=Identity()):
        X = self.X_operator(split_operator * self.field, split_operator * self.regressors if self.sym_regressors != [] else self.regressors)
        return self.filter_yvars_in_Xvars(self.get_y(split_operator), x_field=X)

    def get_y(self, split_operator=Identity()):
        return self.y_operator(split_operator * self.field)

    def get_X_dframe(self, split_operator=Identity()):
        return self.get_X(split_operator).to_pandas()

    def get_y_dframe(self, split_operator=Identity()):
        return self.get_y(split_operator).to_pandas()

    def get_Xy_eq(self):
        X = self.get_X()
        X = SymField([SymVariable(x.name, SymVariable.get_init_info_from_variable(x)[1], x.domain) for x in X.data])
        Y = self.get_y()
        Y = SymField([SymVariable(y.name, SymVariable.get_init_info_from_variable(y)[1], y.domain) for y in Y.data])
        return X, Y


##########################################################
#               StandardScalerForPDE
##########################################################
class StandardScalerForPDE(StandardScaler):
    def sym_var_transform(self, X, y='deprecated', copy=None):
        if isinstance(X, SymVariable):
            X = SymField(X)
        # if isinstance(X, Variable):
        #     X = Field(X)
        if isinstance(X, SymField):
            new_field = X * 1
            if self.with_mean:
                new_field = new_field - self.mean_
            if self.with_std:
                new_field = new_field / self.scale_
            return new_field

    def sym_var_inverse_transform(self, X, copy=None):
        if isinstance(X, SymVariable):
            X = SymField(X)
        if isinstance(X, Variable):
            X = Field(X)
        if isinstance(X, (SymField, Field)):
            new_field = X * 1
            if self.with_std:
                new_field = new_field * self.scale_
            if self.with_mean:
                new_field = new_field + self.mean_
            return new_field


##########################################################
#                       PDEFinder
##########################################################
class PDEFinder:
    def __init__(self, with_mean=True, with_std=True, use_lasso=True):
        self.lasso_cv = None
        self.X_scaler = StandardScalerForPDE(with_mean=with_mean, with_std=with_std)
        self.y_scaler = StandardScalerForPDE(with_mean=with_mean, with_std=with_std)
        self.coefs_ = pd.DataFrame()
        self.coef_threshold = 0
        self.feature_importance = pd.DataFrame()
        self.use_lasso = use_lasso

    def prepare_for_fitting(self, X_train, y_train):
        """
        Creates transformations functions to make fitting more stable. (Standarize)
        :param X_train:
        :param y_train:
        :return:
        """
        self.X_scaler.fit(X_train)
        self.y_scaler.fit(y_train)

        if self.X_scaler.with_mean is False:
            self.X_scaler.mean_ = 0
        if self.y_scaler.with_mean is False:
            self.y_scaler.mean_ = 0

        if self.X_scaler.with_std is False:
            self.X_scaler.scale_ = 1
        if self.y_scaler.with_std is False:
            self.y_scaler.scale_ = 1

    def set_fitting_parameters(self, cv=10, n_alphas=100, alphas=None, max_iter=10000):
        # self.lasso_cv = MultiTaskLassoCV(eps=0.0001,
        #                                  n_alphas=n_alphas,
        #                                  alphas=alphas,
        #                                  fit_intercept=False,
        #                                  normalize=False,
        #                                  # precompute='auto',
        #                                  max_iter=max_iter,
        #                                  tol=0.0001,
        #                                  copy_X=True,
        #                                  cv=cv,
        #                                  verbose=False,
        #                                  n_jobs=-1,
        #                                  # positive=False,
        #                                  random_state=None,
        #                                  selection='cyclic')
        # self.lasso_cv = ElasticNetCV(alphas=alphas,
        #                              copy_X=True,
        #                              cv=cv,
        #                              eps=0.00001,
        #                              fit_intercept=False,
        #                              l1_ratio=0.2,
        #                              max_iter=max_iter,
        #                              n_alphas=n_alphas,
        #                              n_jobs=-1,
        #                              normalize=False,
        #                              positive=False,
        #                              precompute='auto',
        #                              random_state=None,
        #                              selection='random',
        #                              tol=0.00001,
        #                              verbose=0)
        if self.use_lasso:
            self.lasso_cv = LassoCV(eps=0.00001,
                                    n_alphas=n_alphas,
                                    alphas=alphas,
                                    fit_intercept=False,
                                    normalize=False,
                                    precompute='auto',
                                    max_iter=max_iter,
                                    tol=0.000001,
                                    copy_X=True,
                                    cv=cv,
                                    verbose=False,
                                    n_jobs=-1,
                                    positive=False,
                                    random_state=None,
                                    selection='random')
        else:
            self.lasso_cv = LinearRegression(fit_intercept=False,
                                             normalize=False,
                                             n_jobs=-1,
                                             copy_X=True)

    # def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
    #     """
    #     Given the derivatives or polynomial couplings between the interest variables it finds the differential equation.
    #     :param X_train:
    #     :param y_train:
    #     :return:
    #     """
    #     self.prepare_for_fitting(X_train, y_train)  # preprocess, scaling via z-score.
    #     # transform first, then fit
    #     self.lasso_cv.fit(self.X_scaler.transform(X_train.values), self.y_scaler.transform(y_train.values))
    #     # transform coefs to forget the scaling and have the real coeficients.
    #     self.coefs_ = pd.DataFrame(self.lasso_cv.coef_,
    #                                columns=X_train.columns,
    #                                index=y_train.columns)

    # def fit(self, X_train, y_train):
    #     """
    #     Given the derivatives or polynomial couplings between the interest variables it finds the differential equation.
    #     :type X_train: pd.DataFrame
    #     :type y_train: pd.DataFrame
    #     :return:
    #     """
    #     self.prepare_for_fitting(X_train, y_train)  # preprocess, scaling via z-score.
    #     for name, yt in zip(y_train.columns, self.y_scaler.transform(y_train).T):
    #         lasso_cv = copy.deepcopy(self.lasso_cv)
    #         # transform first, then fit
    #         lasso_cv.fit(self.X_scaler.transform(X_train.values), yt)
    #
    #         # transform coefs to forget the scaling and have the real coeficients.
    #         self.coefs_ = pd.concat([self.coefs_, pd.DataFrame(lasso_cv.coef_.reshape(1, -1),
    #                                                            columns=X_train.columns,
    #                                                            index=[name])],
    #                                 axis=0)
    #     self.determine_feature_importance(X_train, y_train)

    def fit(self, X_train, y_train):
        """
        Given the derivatives or polynomial couplings between the interest variables it finds the differential equation.
        :type X_train: pd.DataFrame
        :type y_train: pd.DataFrame
        :return:
        """
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        if X_train.isna().any().any() or y_train.isna().any().any():
            where_nas = X_train.isna().any(axis=1) | y_train.isna().any(axis=1)
            X_train = X_train.loc[~where_nas, :]
            y_train = y_train.loc[~where_nas, :]
            print('Warning, there where {} rows in trainig data with NaN or Infs'.format(np.sum(where_nas)))

        self.prepare_for_fitting(X_train, y_train)  # preprocess, scaling via z-score.
        for name, yt in zip(y_train.columns, self.y_scaler.transform(y_train).T):
            X_train_temp = self.X_scaler.transform(X_train.values)
            # transform first, then fit
            coefs_temp = np.zeros(X_train_temp.shape[1])
            mask = np.repeat(True, X_train_temp.shape[1])
            for i in range(10):  # maximum depth
                lasso_cv = copy.deepcopy(self.lasso_cv)
                lasso_cv.fit(X_train_temp[:, mask], yt)
                coefs_temp[mask] = lasso_cv.coef_

                if np.all(lasso_cv.coef_ != 0) or np.all(
                        lasso_cv.coef_ == 0):  # all the coefs are not zero then is the minimum expression possible
                    break
                mask[mask] = lasso_cv.coef_ != 0

            # transform coefs to forget the scaling and have the real coeficients.
            self.coefs_ = pd.concat([self.coefs_, pd.DataFrame(coefs_temp.reshape(1, -1),
                                                               columns=X_train.columns,
                                                               index=[name])],
                                    axis=0)

        self.coefs_ = self._get_coefs(self.coefs_)
        self.determine_feature_importance(X_train, y_train)

    def determine_feature_importance(self, X, y):
        min_error = error(self.inner_transform(X, self.coefs_), y).values
        max_error = error(self.inner_transform(X, self.coefs_ * 0), y).values

        self.feature_importance = pd.DataFrame(0, columns=self.coefs_.columns, index=self.coefs_.index)
        if np.any(max_error < min_error):
            self.feature_importance = np.nan * self.feature_importance
        else:
            for target_i, j in itertools.product(*list(map(lambda x: list(range(x)), self.coefs_.shape))):
                coefs_ = self.coefs_.copy()
                coefs_.iloc[target_i, j] = 0
                self.feature_importance.iloc[target_i, j] = (error(self.inner_transform(X, coefs_.iloc[target_i, :]),
                                                                   y.iloc[:, target_i]) - min_error[target_i]) / \
                                                            (max_error - min_error)[target_i]
            self.feature_importance = self.feature_importance.divide(self.feature_importance.sum(axis=1), axis=0)

    def _get_coefs(self, coefs):
        coefs_ = coefs / self.X_scaler.scale_
        coefs_ = coefs_.apply(lambda c: c * self.y_scaler.scale_)
        # TODO: if no 1.00000... (the constant) is present this fails. Because it always should have to balance zscoreing
        if any(["1.00" in c for c in coefs_.columns]):
            coefs_.loc[:, [True if "1.00" in c else False for c in coefs_.columns]] += (
                    self.y_scaler.mean_ - np.dot(coefs_, self.X_scaler.mean_)).reshape((-1, 1))

        coefs_[coefs_.abs() < self.coef_threshold] = 0
        return coefs_

    def inner_transform(self, X, coefs):
        return np.matmul(X.values, coefs.values.T)

    def transform(self, X):
        return self.inner_transform(X, self.coefs_)

    def get_equation(self, sym_x, sym_y):
        """

        :param sym_x: SymField
        :param sym_y: SymField
        :return:
        :rtype: (SymVariable, SymField)
        """
        sym_field_res = sym_x.matmul(self.coefs_.T)
        sym_field_res = sym_field_res - sym_y
        sym_field_res.simplify()
        return sym_field_res

    def predict(self, split_data_operator_list, dm, starting_point, domain_variable2predict, horizon):
        """

        :param split_data_operator_list:
        :type dm: DataManager
        :param starting_point: when more than one variable it is used to define the other domain points.
        :type starting_point: dict
        :type domain_variable2predict: str
        :type horizon: int
        :return:
        """
        var_names = [var.get_full_name() for var in dm.field.data]
        df_predictions_list = []
        for split_data_operator in tqdm(split_data_operator_list):
            df_predictions_list.append(pd.DataFrame(np.nan, index=list(range(horizon)), columns=var_names, dtype=float))

            new_dm = DataManager()
            new_dm.add_variables(split_data_operator * dm.field)
            new_dm.add_regressors(split_data_operator * dm.regressors)  # add regressors without splitting
            new_dm.set_X_operator(dm.X_operator)
            new_dm.set_y_operator(dm.y_operator)
            eq = self.get_equation(new_dm.get_X_sym(), new_dm.get_y_sym()).data
            sub_original_field = new_dm.field

            last_function_value = {}
            predictions_dict = OrderedDict()
            next_point = {var_name: starting_point.copy() for var_name in var_names}

            # --------------- first pass to get the starting point ---------------------
            for sym_eq, original_var, var_name in zip(eq, sub_original_field.data, var_names):
                last_function_value[var_name] = original_var.index_eval(next_point[var_name])
                # the resulting lag of the equation f(t+2)-f(t-1)
                backward_lag, forward_lag = get_lag_from_sym_expression(sym_eq.sym_expression)
                # the starting point to have only 1 unknown.
                next_point[var_name][domain_variable2predict] = sym_eq.domain.shape[domain_variable2predict] - \
                                                                forward_lag[domain_variable2predict]
                df_predictions_list[-1].loc[0, var_name] = last_function_value[var_name]

            # ---------------------- through the future and more ------------------------
            for h in tqdm(range(1, horizon + 1)):
                for sym_eq, original_var, var_name in zip(eq, sub_original_field.data, var_names):
                    expression = sym_eq.evaluate(next_point[var_name])
                    regressors_dict = {
                        reg_sym.sym_expression.subs(next_point[var_name]): reg.index_eval(next_point[var_name])
                        for reg_sym, reg in zip(dm.sym_regressors.data, dm.regressors.data)}

                    valsdict = {}
                    valsdict.update(predictions_dict)
                    valsdict.update(regressors_dict)
                    expression = expression.subs(valsdict)
                    # expression = expression.subs({**predictions_dict, **regressors_dict})
                    solution_list = sympy.solve(expression)
                    solution_list = [{k: sympy.functions.re(v) for k, v in solution.items()} for solution in
                                     solution_list]
                    solution_dict = min(solution_list, key=lambda sol_dict: abs(
                        list(sol_dict.values())[0] - last_function_value[var_name]))
                    predictions_dict.update(solution_dict)
                    last_function_value[var_name] = list(solution_dict.values())[0]
                    try:
                        float(last_function_value[var_name])
                    except:
                        raise Exception('Last prediction was not a float.')
                    df_predictions_list[-1].loc[h, var_name] = last_function_value[var_name]
                    next_point[var_name][domain_variable2predict] += 1

        return df_predictions_list

    def integrate(self, split_data_operator_list, dm, starting_point, domain_variable2predict,
                  horizon, method='Euler'):
        """

        :param split_data_operator_list:
        :type dm: DataManager
        :param starting_point: when more than one variable it is used to define the other domain points.
        :type starting_point: dict
        :type domain_variable2predict: str
        :type horizon: int
        :return:
        """
        assert len(dm.domain) == 1, "only works with 1d variables."
        ax_name = dm.domain.axis_names[0]

        eq_x_sym_expression, eq_y_sym_expression = dm.get_Xy_eq()
        eq_x_sym_expression = eq_x_sym_expression.matmul(self.coefs_.T).data[0].sym_expression
        eq_y_sym_expression = eq_y_sym_expression.data[0].sym_expression
        der_atoms = get_sorted_derivative_atoms(eq_x_sym_expression - eq_y_sym_expression)
        ode_func = get_func_for_ode(eq_x_sym_expression, eq_y_sym_expression)

        var_names = [var.get_full_name() for var in dm.field.data]
        df_predictions_list = []
        for split_data_operator in tqdm(split_data_operator_list):
            new_dm = DataManager()
            new_dm.add_variables(split_data_operator * dm.field)
            new_dm.add_regressors(split_data_operator * dm.regressors)  # add regressors without splitting
            new_dm.set_X_operator(dm.X_operator)
            new_dm.set_y_operator(dm.y_operator)
            new_dm.set_domain()
            sub_original_field = new_dm.field
            eq = self.get_equation(new_dm.get_X_sym(), new_dm.get_y_sym()).data

            # --------------- first pass to get the starting point ---------------------
            for sym_eq, original_var, var_name in zip(eq, sub_original_field.data, var_names):
                backward_lag, forward_lag = get_lag_from_sym_expression(sym_eq.sym_expression)
                init_point = starting_point.copy()
                init_point[domain_variable2predict] = sym_eq.domain.shape[domain_variable2predict] - \
                                                      forward_lag[domain_variable2predict]

                v0 = (PolyD(derivative_order_dict={ax_name: len(der_atoms)}) * new_dm.field).evaluate_ix(init_point)
                v0 = [v0[str(f).replace(' ', '')] if i == 0 else v0['1.0*' + str(f).replace(' ', '')] for i, f in
                      enumerate([dm.field.data[0].name] + der_atoms[:-1])]

                t0 = new_dm.domain.get_value_from_index(ax_name, init_point[ax_name])
                t = np.arange(t0,
                              t0 + (forward_lag[domain_variable2predict] + horizon) * new_dm.domain.step_width[ax_name],
                              new_dm.domain.step_width[ax_name])

                # ode_func = get_func_for_ode(eq_x_sym_expression, eq_y_sym_expression)
                solver = getattr(odespy, method)(ode_func)
                # solver = method(ode_func)
                solver.set_initial_condition(v0)
                v, t = solver.solve(t)
                # v = self.integrator_core(method, t, v0, get_func_for_ode, eq_x_sym_expression, eq_y_sym_expression)
                # v = odeint(ode_func, v0, t, hmax=new_dm.domain.step_width["t"], hmin=new_dm.domain.step_width["t"],
                #            h0=new_dm.domain.step_width["t"])
                df_pred = pd.DataFrame(v[-horizon:, 0], index=list(range(horizon)), columns=var_names)
                df_pred = df_pred.astype(float)
                df_predictions_list.append(df_pred)

        return df_predictions_list

    def integrate2(self, dm, dery, starting_point, horizon, method='Euler'):
        """

        :param split_data_operator_list:
        :type dm: DataManager
        :param starting_point: when more than one variable it is used to define the other domain points.
        :type starting_point: dict
        :type domain_variable2predict: str
        :type horizon: int
        :return:
        """
        assert len(dm.domain) == 1, "only works with 1d variables."
        ax_name = dm.domain.axis_names[0]
        var_names = [var.get_full_name() for var in dm.field.data]

        eq_x_sym_expression, eq_y_sym_expression = dm.get_Xy_eq()
        ode_func = get_func_for_ode(eq_x_sym_expression.matmul(self.coefs_.T),
                                    eq_y_sym_expression, dm.regressors)

        split_data_operator = DataSplitIndexClip(axis_start_dict=starting_point, axis_len_dict={ax_name: 2*dery})
        new_dm = DataManager()
        new_dm.add_variables(split_data_operator * dm.field)
        new_dm.add_regressors(split_data_operator * dm.regressors)
        new_dm.set_X_operator(dm.X_operator)
        new_dm.set_y_operator(dm.y_operator)
        new_dm.set_domain()

        init_point = starting_point.copy()
        # get derivatives up to the unknown
        v0 = []
        term_names = []
        for sym_var, var in zip(new_dm.sym_field.data, new_dm.field.data):
            terms = [var.name.diff(ax_name, i) for i in range(dery)]
            v0_temp = (PolyD(derivative_order_dict={ax_name: dery - 1}) * var).evaluate_ix(init_point)
            v0_temp = [v0_temp[str(f).replace(' ', '')] if i == 0 else v0_temp['1.0*' + str(f).replace(' ', '')] for
                       i, f in enumerate(terms)]
            v0 += v0_temp
            term_names += terms

        t0 = new_dm.domain.get_value_from_index(ax_name, init_point[ax_name])
        t = np.arange(t0,
                      t0 + (dery + horizon) * new_dm.domain.step_width[ax_name],
                      new_dm.domain.step_width[ax_name])

        if 'RK4' in method:
            solver = getattr(odespy, 'RK4')(ode_func)
        else:
            raise Exception('Method not odentified')
        solver.set_initial_condition(v0)
        v, t = solver.solve(t)

        if len(v.shape) == 1:
            v = v.reshape((-1, 1))

        # v = odeint(ode_func, v0, t)
        df_pred = pd.DataFrame(v[-horizon:, np.linspace(0, v.shape[1], len(var_names)+1, dtype=int)[:-1]],
                               index=list(range(horizon)), columns=var_names)
        df_pred = df_pred.astype(float)

        return df_pred

    def integrate3(self, dm, t, v0, method='Euler'):
        """

        :param split_data_operator_list:
        :type dm: DataManager
        :param starting_point: when more than one variable it is used to define the other domain points.
        :type starting_point: dict
        :type domain_variable2predict: str
        :type horizon: int
        :return:
        """
        assert len(dm.domain) == 1, "only works with 1d variables."
        var_names = [var.get_full_name() for var in dm.field.data]

        eq_x_sym_expression, eq_y_sym_expression = dm.get_Xy_eq()
        ode_func = get_func_for_ode(eq_x_sym_expression.matmul(self.coefs_.T),
                                    eq_y_sym_expression, dm.regressors)

        # split_data_operator = DataSplitIndexClip(axis_start_dict=starting_point, axis_len_dict={ax_name: 2*dery})
        # new_dm = DataManager()
        # new_dm.add_variables(split_data_operator * dm.field)
        # new_dm.add_regressors(split_data_operator * dm.regressors)
        # new_dm.set_X_operator(dm.X_operator)
        # new_dm.set_y_operator(dm.y_operator)
        # new_dm.set_domain()
        #
        # init_point = starting_point.copy()
        # # get derivatives up to the unknown
        # v0 = []
        # term_names = []
        # for sym_var, var in zip(new_dm.sym_field.data, new_dm.field.data):
        #     terms = [var.name.diff(ax_name, i) for i in range(dery)]
        #     v0_temp = (PolyD(derivative_order_dict={ax_name: dery - 1}) * var).evaluate_ix(init_point)
        #     v0_temp = [v0_temp[str(f).replace(' ', '')] if i == 0 else v0_temp['1.0*' + str(f).replace(' ', '')] for
        #                i, f in enumerate(terms)]
        #     v0 += v0_temp
        #     term_names += terms
        #
        # t0 = new_dm.domain.get_value_from_index(ax_name, init_point[ax_name])
        # t = np.arange(t0,
        #               t0 + (dery + horizon) * new_dm.domain.step_width[ax_name],
        #               new_dm.domain.step_width[ax_name])

        if 'RK4' in method:
            solver = getattr(odespy, 'RK4')(ode_func)
        else:
            raise Exception('Method not odentified')
        solver.set_initial_condition(v0)
        v, t = solver.solve(t)

        if len(v.shape) == 1:
            v = v.reshape((-1, 1))

        # v = odeint(ode_func, v0, t)
        df_pred = pd.DataFrame(v[-len(t):, np.linspace(0, v.shape[1], len(var_names)+1, dtype=int)[:-1]],
                               index=t, columns=var_names)
        df_pred = df_pred.astype(float)

        return df_pred
