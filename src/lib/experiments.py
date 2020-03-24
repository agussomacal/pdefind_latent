#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import sys
# import yaml
# import yaml
from datetime import datetime

import numpy as np
import pandas as pd
# to avoid error:
#     self.tk.call('image', 'delete', self.name)
# RuntimeError: main thread is not in main loop
# Tcl_AsyncDelete: async handler deleted by the wrong thread
# import matplotlib
# matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler

import matplotlib.pylab as plt
import seaborn as sns
import os

src_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(src_path)

from src.lib.pdefind import PDEFinder, DataManager
from src.lib.variables import Variable, Domain
from src.lib.operators import PolyD, D, Poly, DataSplit, Delay, MultipleDelay, Identity, DataSplitIndexClip, \
    DataSplitOnIndex
import src.lib.evaluators as evaluator
from src.scripts.utils import evaluate_predictions, savefig, save_csv, load_csv, save, load, varname2latex
from src.scripts import config


# ---------- dictionary of functions and target ----------
def get_x_operator_func(derivative_order, polynomial_order, rational=False):
    def x_operator(field, regressors):
        new_field = copy.deepcopy(field)
        if derivative_order > 0:
            new_field = PolyD({'t': derivative_order}) * new_field
        new_field.append(regressors)
        if rational:
            new_field.append(new_field.__rtruediv__(1.0))
        new_field = Poly(polynomial_order) * new_field
        return new_field

    return x_operator


def get_y_operator_func(derivative_order_y):
    def y_operator(field):
        new_field = D(derivative_order_y, "t") * field
        return new_field

    return y_operator


def get_x_operator_func_delay(delay_order, polynomial_order, rational=False):
    def x_operator(field):
        new_field = copy.deepcopy(field)
        if delay_order > 0:
            new_field = MultipleDelay({'t': list(range(1, delay_order + 1))}) * new_field
        new_field1 = (Poly(polynomial_order) * new_field)
        if rational:
            new_field2 = Poly(polynomial_order) * new_field.__rtruediv__(1.0)
        else:
            new_field2 = []
        new_field1.append(new_field2)
        return new_field1

    return x_operator


class ExperimentSetting:
    def __init__(self, experiment_name):
        # ---------------- paths --------------------
        self.experiment_name = experiment_name
        self.plots_path = config.plots_dir + experiment_name + "/"
        if not os.path.exists(self.plots_path):
            os.mkdir(self.plots_path)

        # ----- pdfind params -----
        self.alphas = None
        self.alpha = None
        self.max_iter = None
        self.cv = None
        self.with_mean = None
        self.with_std = None
        self.max_num_params = np.inf
        self.max_train_cases = np.inf

        # ----- plot params -----
        self.sub_set_len = None
        self.sub_set_init = None
        self.colors = ['blue', 'green', 'red', 'purple', 'orange', 'black', 'magenta', 'cyan']

        # ----- regressors_builders -----
        self.regressors_builders = []

        # ----- split data params -----
        self.trainSplit = None
        self.testSplit = None

        # ----- prediction params -----
        self.horizon = 0
        self.num_evaluations = 0

        # ----- phase_diagram_horizon params -----
        self.phase_diagram_horizon = 0

        # ----- experiments record -----
        self.experiments = []

    # ========================= setters =========================
    def generate_params_file(self):
        params2save = {k: v for k, v in self.__dict__.items() if k not in ['data', 't', 't0']}

        # get previous experiments
        if os.path.exists(config.get_filename('params.yml', self.experiment_name)):
            with open(config.get_filename('params.yml', self.experiment_name), 'r') as outfile:
                params2save['experiments'].append(yaml.load(outfile)['experiments'])

        with open(config.get_filename('params.yml', self.experiment_name), 'w') as outfile:
            yaml.dump(params2save, outfile, default_flow_style=False)

    # ========================= setters =========================
    def set_pdefind_params(self, with_mean=True, with_std=True, alphas=100, max_iter=10000, cv=20, alpha=None,
                           max_num_params=np.inf, max_train_cases=np.inf, use_lasso=True):
        self.alphas = alphas
        self.alpha = alpha
        self.max_iter = max_iter
        self.cv = cv
        self.with_mean = with_mean
        self.with_std = with_std
        self.max_num_params = max_num_params
        self.max_train_cases = max_train_cases
        self.use_lasso = use_lasso

    def set_plotting_params(self, sub_set_init, sub_set_len):
        self.sub_set_len = sub_set_len
        self.sub_set_init = sub_set_init

    def set_data_split(self, trainSplit=DataSplit({"t": 0.7}), testSplit=DataSplit({"t": 0.3}, {"t": 0.7})):
        self.trainSplit = trainSplit
        self.testSplit = testSplit

    def set_regressor_builders(self, regressors_builders):
        self.regressors_builders = regressors_builders

    def set_prediction_params(self, horizon, num_evaluations):
        self.horizon = horizon
        self.num_evaluations = num_evaluations

    def set_phase_diagram_params(self, horizon):
        self.phase_diagram_horizon = horizon

    # ========================= get data manager =========================
    def get_domain(self):
        pass

    def get_variables(self):
        return []

    def get_regressors(self):
        # TODO: only works with time variable regressor
        reggressors = []
        domain = self.get_domain()
        variables = self.get_variables()
        for reg_builder in self.regressors_builders:
            for variable in variables:
                reg_builder.fit(self.trainSplit * variable.domain, self.trainSplit * variable)
                serie = reg_builder.transform(
                    domain.get_range(axis_names=[reg_builder.domain_axes_name])[reg_builder.domain_axes_name])
                reggressors.append(Variable(serie, domain, domain2axis={reg_builder.domain_axes_name: 0},
                                            variable_name='{}_{}'.format(variable.get_name(), reg_builder.name)))
        return reggressors

    def get_data_manager(self):
        data_manager = DataManager()
        data_manager.add_variables(self.get_variables())
        data_manager.add_regressors(self.get_regressors())
        data_manager.set_domain()
        return data_manager

    # ========================= fitting model =========================

    def fit_eqdifff(self, data_manager):
        with evaluator.timeit('pdefind fitting'):
            pde_finder = PDEFinder(with_mean=self.with_mean, with_std=self.with_std, use_lasso=self.use_lasso)
            pde_finder.set_fitting_parameters(cv=self.cv, n_alphas=self.alphas, max_iter=self.max_iter,
                                              alphas=self.alpha)
            X = data_manager.get_X_dframe(self.trainSplit)
            Y = data_manager.get_y_dframe(self.trainSplit)

            if X.shape[0] > self.max_train_cases:
                sample = np.random.choice(X.shape[0], size=self.max_train_cases)
                X = X.iloc[sample, :]
                Y = Y.iloc[sample, :]
            # if X.shape[1] > self.max_num_params:
            #     raise Exception('More params than allowed: params of X={} and max value of params is {}'.format(
            #         X.shape[1], self.max_num_params))
            pde_finder.fit(X, Y)
        return pde_finder

    def load_fitsave_eqdifff(self, data_manager, filename=None, subfolders=None):
        pde_finder = load(filename + '.pickle', self.experiment_name, subfolders)
        if not pde_finder:
            save(self.fit_eqdifff(data_manager), filename + '.pickle', self.experiment_name, subfolders)
        return pde_finder

    # ========================= auxiliar functions =========================
    def get_test_time(self, data_manager, type='Variable'):
        t = Variable(data_manager.domain.get_range('t')['t'], data_manager.domain, domain2axis={'t': 0},
                     variable_name='t')
        t = self.testSplit * t
        if type == 'Variable':
            return t
        elif type == 'numpy':
            return np.array(t.data)
        else:
            raise Exception('Not implemented return type: only Variable and numpy')

    def get_test_y_yhat(self, pde_finder, data_manager):
        y = data_manager.get_y_dframe(self.testSplit)
        yhat = pd.DataFrame(pde_finder.transform(data_manager.get_X_dframe(self.testSplit)), columns=y.columns)
        return y, yhat

    def get_rsquare_of_eqdiff_fit(self, pde_finder, data_manager):
        y, yhat = self.get_test_y_yhat(pde_finder, data_manager)
        return evaluator.rsquare(yhat=yhat, y=y)

    @staticmethod
    def get_derivative_in_y(derivative_in_y, derivatives_in_x):
        # -1 means the 1 more from the depth
        if derivative_in_y == -1:
            der_y = derivatives_in_x + 1
        else:
            der_y = derivative_in_y
        return der_y

    # ========================= plots =========================
    def plot_fitted_vs_real(self, pde_finder, data_manager, col="blue"):
        y, yhat = self.get_test_y_yhat(pde_finder, data_manager)

        plt.close('all')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        ax.plot(y, yhat, '.', c=col, alpha=0.7)
        ax.set_xlabel("real data points")
        ax.set_ylabel("fitted data points")
        ax.set_title("Fitted vs Real")
        return ax

    @staticmethod
    def plot_phase_diagram(x, dx, method, var_name, color, ax):
        dx = np.gradient(x) / dx

        ax.plot(x[1:-1], dx[1:-1], '-', c=color, alpha=0.7, label=method, linewidth=2)
        ax.set_xlabel(varname2latex(var_name), fontsize=20)
        ax.set_ylabel(varname2latex(var_name, derivative=1), fontsize=20, rotation=0)
        ax.set_title('Phase diagram')
        ax.legend(fontsize=20)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        return np.array(x[1:-1]).reshape((-1, 1)), np.array(dx[1:-1]).reshape((-1, 1))

    def plot_fitted_and_real(self, pde_finder, data_manager, col="blue", subinit=None, sublen=None):
        y, yhat = self.get_test_y_yhat(pde_finder, data_manager)
        t = self.get_test_time(data_manager, type='numpy')

        if subinit is not None:
            y = y[subinit:]
            yhat = yhat[subinit:]
            t = t[subinit:]
        if sublen is not None:
            y = y[:sublen]
            yhat = yhat[:sublen]
            t = t[:sublen]

        plt.close('all')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        ax.plot(t, yhat, '.-', c=col, label="fitted", alpha=0.45)
        ax.plot(t, y, 'k-', label="real", alpha=0.7)
        ax.set_xlabel("time")
        ax.set_ylabel("function")
        ax.set_title("Time series: real and fitted")
        ax.legend()

    def plot_feature_importance(self, pdefind):
        pdefind.feature_importance.T.plot.barh(rot=45, legend=False)
        plt.tight_layout()

    def plot_coefficients(self, pdefind):
        dfc = pdefind.coefs_.T
        dfc.index = [
            c.replace('1.0', '').replace('0', '').replace('Derivative', 'D').replace('**', '^').replace('*', ' ') for c
            in dfc.index]
        dfc.plot.barh(legend=True)
        plt.tight_layout()

    # ========================= explore polynomial and derivatives =========================
    def explore_eqdiff_fitting(self, derivative_in_y, derivatives2explore, poly2explore, rational=False, getXfunc=get_x_operator_func):
        # ---------- save params of experiment ----------
        self.experiments.append({'explore_eqdiff_fitting': {
            'date': datetime.now(),
            'derivative_in_y': derivative_in_y,
            'derivatives2explore': derivatives2explore,
            'poly2explore': poly2explore}
        })

        rsquares = pd.DataFrame(np.nan, columns=poly2explore, index=derivatives2explore)
        for poly_degree in poly2explore:
            print("\n---------------------")
            print("Polynomial degree: {}".format(poly_degree))
            print "Derivative order:",

            for derivative_depth in derivatives2explore:
                print " {}".format(derivative_depth),
                data_manager = self.get_data_manager()
                data_manager.set_X_operator(getXfunc(derivative_depth, poly_degree, rational=rational))
                data_manager.set_y_operator(
                    get_y_operator_func(self.get_derivative_in_y(derivative_in_y, derivative_depth)))

                pde_finder = self.fit_eqdifff(data_manager)
                rsquares.loc[derivative_depth, poly_degree] = np.mean(self.get_rsquare_of_eqdiff_fit(pde_finder,
                                                                                                     data_manager).values)

                subname = '_y{}_der_x{}_pol{}'.format(derivative_in_y, derivative_depth, poly_degree)
                # with savefig('feature_importance_der{}'.format(subname), self.experiment_name,
                #              subfolders=['derivative_in_y_{}'.format(derivative_in_y), 'feature_importances']):
                #     self.plot_feature_importance(pde_finder)

                with savefig('fit_vs_real_{}'.format(subname),
                             self.experiment_name,
                             subfolders=['derivative_in_y_{}'.format(derivative_in_y), 'fit_vs_real']):
                    self.plot_fitted_vs_real(pde_finder, data_manager)

                with savefig('fit_and_real_{}'.format(subname),
                             self.experiment_name,
                             subfolders=['derivative_in_y_{}'.format(derivative_in_y), 'fit_and_real']):
                    self.plot_fitted_and_real(pde_finder, data_manager)

                with savefig('zoom_fit_and_real{}'.format(subname),
                             self.experiment_name,
                             subfolders=['derivative_in_y_{}'.format(derivative_in_y), 'fit_and_real_zoom']):
                    self.plot_fitted_and_real(pde_finder, data_manager, subinit=self.sub_set_init,
                                              sublen=self.sub_set_len)

        if derivative_in_y == -1:
            # because we want to plot the maximum derivative value.
            rsquares.index = rsquares.index + 1
        save_csv(rsquares, 'rsquares_eqfit_der_y{}_rational{}'.format(derivative_in_y, rational), self.experiment_name)

        # ---------- plot heatmap of rsquares ----------
        with savefig('rsquares_eqfit_der_y{}_rational{}'.format(derivative_in_y, rational), self.experiment_name):
            plt.close('all')
            sns.heatmap(rsquares * (rsquares > 0), annot=True)
            plt.xlabel("Polynomial max order")
            plt.ylabel("Derivative max order")
            plt.title("rsquares for derivative in y {}".format(derivative_in_y))

    # ========================= explore polynomial and delays =========================
    def explore_delay(self, delays2explore, poly2explore, derivative_in_y=0, rational=False):
        # ---------- save params of experiment ----------
        self.experiments.append({'explore_eqdiff_fitting': {
            'date': datetime.now(),
            'derivatives2explore': delays2explore,
            'poly2explore': poly2explore}
        })

        rsquares = pd.DataFrame(np.nan, columns=poly2explore, index=delays2explore)
        for poly_degree in poly2explore:
            print("\n---------------------")
            print("Polynomial degree: {}".format(poly_degree))
            print "Delay order:",

            for delay_in_y in delays2explore:
                print " {}".format(delay_in_y),
                data_manager = self.get_data_manager()
                data_manager.set_X_operator(get_x_operator_func_delay(delay_in_y, poly_degree, rational=rational))
                data_manager.set_y_operator(lambda field: Delay(delay=derivative_in_y, axis_name='t') * field)

                pde_finder = self.fit_eqdifff(data_manager)
                rsquares.loc[delay_in_y, poly_degree] = self.get_rsquare_of_eqdiff_fit(pde_finder, data_manager).values

                subname = 'del_y_{}_der_x{}_pol{}'.format(derivative_in_y, delay_in_y, poly_degree)
                with savefig('feature_importance_der{}'.format(subname), self.experiment_name,
                             subfolders=['derivative_in_y_{}'.format(derivative_in_y), 'feature_importances']):
                    self.plot_feature_importance(pde_finder)

                with savefig('fit_vs_real_{}'.format(subname),
                             self.experiment_name,
                             subfolders=['derivative_in_y_{}'.format(derivative_in_y), 'fit_vs_real']):
                    self.plot_fitted_vs_real(pde_finder, data_manager)

                with savefig('fit_and_real_{}'.format(subname),
                             self.experiment_name,
                             subfolders=['derivative_in_y_{}'.format(derivative_in_y), 'fit_and_real']):
                    self.plot_fitted_and_real(pde_finder, data_manager)

                with savefig('zoom_fit_and_real{}'.format(subname),
                             self.experiment_name,
                             subfolders=['derivative_in_y_{}'.format(derivative_in_y), 'fit_and_real_zoom']):
                    self.plot_fitted_and_real(pde_finder, data_manager, subinit=self.sub_set_init,
                                              sublen=self.sub_set_len)

        if derivative_in_y == -1:
            # because we want to plot the maximum derivative value.
            rsquares.index = rsquares.index + 1
        save_csv(rsquares, 'rsquares_eqfit_der_y{}_rational{}'.format(derivative_in_y, rational), self.experiment_name)

        # ---------- plot heatmap of rsquares ----------
        with savefig('rsquares_eqfit_der_y{}_rational{}'.format(derivative_in_y, rational), self.experiment_name):
            plt.close('all')
            sns.heatmap(rsquares * (rsquares > 0), annot=True)
            plt.xlabel("Polynomial max order")
            plt.ylabel("Derivative max order")
            plt.title("rsquares for derivative in y {}".format(derivative_in_y))

    # ========================= noise evaluation =========================
    def explore_noise_discretization(self, noise_range, discretization_range, derivative_in_y, derivatives_in_x,
                                     poly_degree, std_of_discrete_grad=False):
        """

        :param noise_range:
        :param discretization_range:
        :param derivative_in_y:
        :param derivatives_in_x:
        :param poly_degree:
        :param std_of_discrete_grad: True if we wan to calculate the std of the gradient of the series using the discretized version. Otherwise will be the original one.
        :return:
        """
        # ---------- save params of experiment ----------
        self.experiments.append({'explore_noise_discretization': {
            'date': datetime.now(),
            'noise_range': noise_range,
            'discretization_range': discretization_range,
            'derivative_in_y': derivative_in_y,
            'derivatives_in_x': derivatives_in_x,
            'poly_degree': poly_degree}
        })

        # ----------------------------------------
        rsquares = pd.DataFrame(np.nan, index=noise_range, columns=discretization_range)
        rsquares.index.name = "Noise"
        rsquares.columns.name = "Discretization"

        # ----------------------------------------
        data_manager = self.get_data_manager()

        std_of_vars = []
        for var in data_manager.field.data:
            series_grad = np.abs(np.gradient(var.data))
            std_of_vars.append(np.std(series_grad))
            with savefig('Distribution_series_differences_{}'.format(var.get_full_name()), self.experiment_name,
                         subfolders=['noise_derivative_in_y_{}'.format(derivative_in_y)]):
                sns.distplot(series_grad, bins=int(np.sqrt(len(var.data))))
                plt.axvline(x=std_of_vars[-1])

        # ---------- Noise evaluation ----------
        for measure_dt in discretization_range:
            print("\n---------------------")
            print("meassure dt: {}".format(measure_dt))
            print "Noise: ",
            for noise in noise_range:
                print noise,
                # choose steps with bigger dt; and sum normal noise.
                new_t = data_manager.domain.get_range("t")['t'][::measure_dt]
                domain_temp = Domain(lower_limits_dict={"t": np.min(new_t)},
                                     upper_limits_dict={"t": np.max(new_t)},
                                     step_width_dict={"t": data_manager.domain.step_width['t'] * measure_dt})

                data_manager_temp = DataManager()
                data_manager_original_temp = DataManager()
                for std, var in zip(std_of_vars, data_manager.field.data):
                    data_original = var.data[::measure_dt]

                    if std_of_discrete_grad:
                        series_grad = np.abs(np.gradient(data_original))
                        std = np.std(series_grad)

                    data = data_original + np.random.normal(loc=0, scale=std * noise, size=len(data_original))
                    data_manager_temp.add_variables(
                        Variable(data, domain_temp, domain2axis={"t": 0}, variable_name=var.name))
                    data_manager_original_temp.add_variables(
                        Variable(data_original, domain_temp, domain2axis={"t": 0}, variable_name=var.name))
                data_manager_temp.add_regressors([])
                data_manager_temp.set_domain()
                data_manager_original_temp.add_regressors([])
                data_manager_original_temp.set_domain()

                data_manager_temp.set_X_operator(get_x_operator_func(derivatives_in_x, poly_degree))
                data_manager_temp.set_y_operator(
                    get_y_operator_func(self.get_derivative_in_y(derivative_in_y, derivatives_in_x)))
                data_manager_original_temp.set_X_operator(get_x_operator_func(derivatives_in_x, poly_degree))
                data_manager_original_temp.set_y_operator(
                    get_y_operator_func(self.get_derivative_in_y(derivative_in_y, derivatives_in_x)))

                pde_finder = self.fit_eqdifff(data_manager_temp)

                y = data_manager_original_temp.get_y_dframe(self.testSplit)
                yhat = pd.DataFrame(pde_finder.transform(data_manager_temp.get_X_dframe(self.testSplit)),
                                    columns=y.columns)
                rsquares.loc[noise, measure_dt] = evaluator.rsquare(yhat=yhat, y=y).values
                # rsquares.loc[noise, measure_dt] = self.get_rsquare_of_eqdiff_fit(pde_finder, data_manager_temp).values

                with savefig('fit_vs_real_der_y{}_noise{}_dt{}'.format(derivative_in_y, str(noise).replace('.', ''),
                                                                       measure_dt),
                             self.experiment_name,
                             subfolders=['noise_derivative_in_y_{}'.format(derivative_in_y), 'fit_vs_real']):
                    self.plot_fitted_vs_real(pde_finder, data_manager_temp)

                with savefig('fit_and_real_der_y{}_noise{}_dt{}'.format(derivative_in_y, str(noise).replace('.', ''),
                                                                        measure_dt),
                             self.experiment_name,
                             subfolders=['noise_derivative_in_y_{}'.format(derivative_in_y), 'fit_and_real']):
                    self.plot_fitted_and_real(pde_finder, data_manager_temp)

                with savefig('zoom_fit_and_real_der_y{}_dt{}_noise{}'.format(derivative_in_y, measure_dt,
                                                                             str(noise).replace('.', '')),
                             self.experiment_name,
                             subfolders=['noise_derivative_in_y_{}'.format(derivative_in_y), 'fit_and_real_zoom']):
                    self.plot_fitted_and_real(pde_finder, data_manager_temp, subinit=self.sub_set_init,
                                              sublen=self.sub_set_len)

        save_csv(rsquares, 'noise_discretization_rsquares_eqfit_der_y{}'.format(derivative_in_y), self.experiment_name)
        # plt.pcolor(rsquares * (rsquares > 0), cmap='autumn')
        # plt.yticks(np.arange(0.5, len(rsquares.index), 1), np.round(rsquares.index, decimals=2))
        # plt.xticks(np.arange(0.5, len(rsquares.columns), 1), rsquares.columns)

        # ---------- plot heatmap of rsquares ----------
        with savefig('noise_discretization_rsquares_eqfit_der_y{}'.format(derivative_in_y), self.experiment_name):
            rsquares.index = np.round(rsquares.index, decimals=2)
            plt.close('all')
            sns.heatmap(rsquares * (rsquares > 0), annot=True)
            plt.xlabel("Discretization (dt)")
            plt.ylabel("Noise (k*std)")
            plt.title("Noise and discretization for derivative in y {}".format(derivative_in_y))

        return rsquares

    # ================== Predictions ================== #
    def do_predictions(self, prediction_methods, pde_finder, data_manager, horizon, num_evaluations, dery,
                       predictions=None):
        if predictions is not None:
            already_made_methods = predictions.method.unique()
            pred_methods = set(prediction_methods).difference(already_made_methods)
        else:
            pred_methods = prediction_methods

        print("\t predictions")
        real, predictions_new = evaluate_predictions(pde_finder,
                                                     data_split_operator=self.testSplit,
                                                     dm=data_manager,
                                                     starting_point={"t": 0},
                                                     domain_variable2predict="t",
                                                     horizon=horizon,
                                                     num_evaluations=num_evaluations,
                                                     dery=dery,
                                                     prediction_methods=pred_methods,
                                                     seed=42)
        return real, pd.concat([predictions, predictions_new])

    def explore_predictions(self, prediction_methods, derivative_in_y, derivatives_in_x, poly_degree,
                            method_label_dict={}, getXfunc=get_x_operator_func):
        # ---------- save params of experiment ----------
        self.experiments.append({'explore_predictions': {
            'date': datetime.now(),
            'prediction_methods': prediction_methods,
            'derivative_in_y': derivative_in_y,
            'derivatives_in_x': derivatives_in_x,
            'poly_degree': poly_degree}
        })

        # ----------------------------------------
        prediction_methods = list(sorted(prediction_methods))

        data_manager = self.get_data_manager()
        data_manager.set_X_operator(getXfunc(derivatives_in_x, poly_degree))
        data_manager.set_y_operator(get_y_operator_func(self.get_derivative_in_y(derivative_in_y, derivatives_in_x)))

        # ---------- fit eqdiff ----------
        pde_finder = self.fit_eqdifff(data_manager)

        # ---------- predictions ----------
        subfolders = ['predictions']
        predictions = load_csv('future_predictions_data', self.experiment_name, subfolders=subfolders)
        real, predictions = self.do_predictions(prediction_methods=prediction_methods,
                                                pde_finder=pde_finder,
                                                dery=derivatives_in_x + 1,
                                                data_manager=data_manager,
                                                horizon=self.horizon,
                                                num_evaluations=self.num_evaluations,
                                                predictions=predictions)
        save_csv(real, 'future_real_data', self.experiment_name, subfolders=subfolders)
        save_csv(predictions, 'future_predictions_data', self.experiment_name, subfolders=subfolders)

        # if we want to append new methods.
        prediction_methods = predictions.method.unique()
        method_colors = {pred_method: self.colors[i] for i, pred_method in enumerate(prediction_methods)}

        if method_label_dict == {}:
            method_label_dict = {method: method for method in prediction_methods}

        # ---------- evaluate statistics ----------
        for var in data_manager.field.data:
            var_name = var.get_full_name()

            # ---------- statistics ----------
            rsq = pd.DataFrame(np.nan, columns=prediction_methods, index=np.arange(self.horizon))
            mape = pd.DataFrame(np.nan, columns=prediction_methods, index=np.arange(self.horizon))
            mape_std = pd.DataFrame(np.nan, columns=prediction_methods, index=np.arange(self.horizon))

            # check if there are methods already calculated
            old_rsq = load_csv('r2_predictions_{}'.format(var_name), self.experiment_name, subfolders=subfolders)
            old_mape = load_csv('mape_predictions_{}'.format(var_name), self.experiment_name, subfolders=subfolders)
            if old_rsq is None:
                old_methods = []
            else:
                old_methods = old_rsq.columns
                rsq[old_methods] = old_rsq
                mape[old_methods] = old_mape

            # calculate statistics
            for method, df in predictions.groupby('method'):
                if method in old_methods:
                    continue

                for (ix_p, dfp), (ix_r, dfr) in zip(df.groupby(level='index'), real.groupby(level='index')):
                    assert ix_p == ix_r
                    rsq.loc[ix_p, method] = evaluator.rsquare(dfp[var_name], dfr[var_name])
                    mape.loc[ix_p, method] = evaluator.mape(dfp[var_name], dfr[var_name])
                    mape_std.loc[ix_p, method] = evaluator.mape_sd(dfp[var_name], dfr[var_name])

            # save
            save_csv(rsq, 'r2_predictions_{}'.format(var_name), self.experiment_name, subfolders=subfolders)
            save_csv(mape, 'mape_predictions_{}'.format(var_name), self.experiment_name, subfolders=subfolders)

            # ---------- plot statistics ----------
            with savefig('R2_{}'.format(var_name), self.experiment_name, subfolders=subfolders):
                fig, ax = plt.subplots()
                for method in rsq.columns:
                    ax.plot(rsq.index[rsq[method] > 0], rsq[method][rsq[method] > 0], '.-', c=method_colors[method],
                            label=method_label_dict[method])
                plt.legend()

            with savefig('mape_{}'.format(var_name), self.experiment_name, subfolders=subfolders):
                fig, ax = plt.subplots()
                for method in rsq.columns:
                    ax.plot(mape.index[mape[method] < 1], mape[method][mape[method] < 1], c=method_colors[method],
                            label=method_label_dict[method])
                    ax.fill_between(mape.index[mape[method] < 1],
                                    mape[method][mape[method] < 1] - mape_std[method][mape[method] < 1],
                                    mape[method][mape[method] < 1] + mape_std[method][mape[method] < 1],
                                    color=method_colors[method], alpha=0.4)
                plt.legend()
            plt.close("all")

    # ================== Predictions ================== #
    # def explore_phase_diagram(self, prediction_methods, derivative_in_y, derivatives_in_x, poly_degree, rational=False,
    #                           method_label_dict={}, reload=True):
    #     # ---------- save params of experiment ----------
    #     self.experiments.append({'explore_phase_diagram': {
    #         'date': datetime.now(),
    #         'prediction_methods': prediction_methods,
    #         'derivative_in_y': derivative_in_y,
    #         'derivatives_in_x': derivatives_in_x,
    #         'poly_degree': poly_degree}
    #     })
    #
    #     subfolders = ['phase_diagram']
    #
    #     # ----------------------------------------
    #     prediction_methods = list(sorted(prediction_methods))
    #
    #     data_manager = self.get_data_manager()
    #     data_manager.set_X_operator(get_x_operator_func(derivatives_in_x, poly_degree, rational))
    #     data_manager.set_y_operator(get_y_operator_func(self.get_derivative_in_y(derivative_in_y, derivatives_in_x)))
    #
    #     # ---------- fit eqdiff ----------
    #     pde_finder = self.fit_eqdifff(data_manager, delay=derivative_in_y)
    #     with savefig('coeficients_{}_dery{}_derx{}_poly{}'.format('_'.join(prediction_methods), derivative_in_y,
    #                                                               derivatives_in_x, poly_degree), self.experiment_name,
    #                  subfolders=subfolders):
    #         self.plot_coefficients(pde_finder)
    #
    #     # ---------- predictions ----------
    #     predictions = load_csv('phase_diagram_predictions_data', self.experiment_name, subfolders=subfolders)
    #     if predictions is not None and not reload and predictions.method.isin(prediction_methods).any():
    #         predictions = predictions.loc[~predictions.method.isin(prediction_methods), :]
    #
    #     df_predictions_list = [pde_finder.integrate2(dm=data_manager,
    #                                                 dery=derivatives_in_x-derivative_in_y if derivative_in_y < 0 else derivative_in_y,
    #                                                 starting_point={"t": 0},
    #                                                 horizon=self.phase_diagram_horizon,
    #                                                 method=prediction_method) for prediction_method in prediction_methods]
    #
    #     predictions = pd.concat([predictions] + df_predictions_list)
    #
    #     # real, predictions = self.do_predictions([], pde_finder, data_manager,
    #     #                                         self.phase_diagram_horizon,
    #     #                                         dery=derivative_in_y,
    #     #                                         num_evaluations=1,
    #     #                                         predictions=predictions)
    #     print(predictions)
    #     # save_csv(real, 'phase_diagram_real_data', self.experiment_name, subfolders=subfolders)
    #     save_csv(predictions, 'phase_diagram_predictions_data', self.experiment_name, subfolders=subfolders)
    #
    #     # if we want to append new methods.
    #     prediction_methods = predictions.method.unique()
    #     method_colors = {pred_method: self.colors[i] for i, pred_method in enumerate(prediction_methods)}
    #
    #     # ---------- evaluate statistics ----------
    #     for var in data_manager.field.data:
    #         var_name = var.get_full_name()
    #
    #         # ---------- plot phase diagram ----------
    #         with savefig('phase_diagram_{}'.format(var_name), self.experiment_name,
    #                      subfolders=subfolders):
    #             fig, ax = plt.subplots()
    #             # self.plot_phase_diagram(real[var_name].values.ravel(), dx=data_manager.domain.step_width['t'],
    #             #                         method='real', var_name=var_name, color='black', ax=ax)
    #             for method, df in predictions.groupby('method'):
    #                     self.plot_phase_diagram(df[var_name].values.ravel(), dx=data_manager.domain.step_width['t'],
    #                                             method=method, var_name=var_name,
    #                                             color=method_colors[method], ax=ax)
    #             plt.legend()
    #
    #         plt.close("all")

    # ================== Predictions ================== #
    def explore_phase_diagram_delayed(self, prediction_methods, max_delay_inl_x, poly_degree, delay_in_y=0,
                                      rational=False, getXfunc=get_x_operator_func_delay):
        # ---------- save params of experiment ----------
        self.experiments.append({'explore_phase_diagram': {
            'date': datetime.now(),
            'prediction_methods': prediction_methods,
            'delay_in_y': delay_in_y,
            'max_delay_in_x': max_delay_in_x,
            'poly_degree': poly_degree}
        })

        # ----------------------------------------
        prediction_methods = list(sorted(prediction_methods))

        data_manager = self.get_data_manager()
        data_manager.set_X_operator(getXfunc(max_delay_in_x, poly_degree, rational))
        data_manager.set_y_operator(lambda field: Delay(axis_name='t', delay=delay_in_y) * field)

        # ---------- fit eqdiff ----------
        pde_finder = self.fit_eqdifff(data_manager)

        # ---------- predictions ----------
        subfolders = ['phase_diagram']
        predictions = load_csv('phase_diagram_predictions_data', self.experiment_name, subfolders=subfolders)
        real, predictions = self.do_predictions(prediction_methods, pde_finder, data_manager,
                                                self.phase_diagram_horizon, num_evaluations=1,
                                                predictions=predictions)
        print(predictions)
        save_csv(real, 'phase_diagram_real_data', self.experiment_name, subfolders=subfolders)
        save_csv(predictions, 'phase_diagram_predictions_data', self.experiment_name, subfolders=subfolders)

        # if we want to append new methods.
        prediction_methods = predictions.method.unique()
        method_colors = {pred_method: self.colors[i] for i, pred_method in enumerate(prediction_methods)}

        # ---------- evaluate statistics ----------
        for var in data_manager.field.data:
            var_name = var.get_full_name()

            # ---------- plot phase diagram ----------
            with savefig('phase_diagram_{}_real'.format(var_name), self.experiment_name, subfolders=subfolders):
                self.plot_phase_diagram(real[var_name].values.ravel(), dx=data_manager.domain.step_width['t'],
                                        method='real', var_name=var_name, color='black')

            for method, df in predictions.groupby('method'):
                with savefig('phase_diagram_{}_{}'.format(var_name, method), self.experiment_name,
                             subfolders=subfolders):
                    self.plot_phase_diagram(df[var_name].values.ravel(), dx=data_manager.domain.step_width['t'],
                                            method=method, var_name=var_name,
                                            color=method_colors[method])

            plt.close("all")

    # ================== Predictions ================== #
    def explore_phase_diagram(self, prediction_methods, derivative_in_y, derivatives_in_x, poly_degree, rational=False,
                              method_label_dict={}, reload=True, starting_point={"t": 0}, prediction_methods2plot=None,
                              getXfunc=get_x_operator_func):
        # ---------- save params of experiment ----------
        self.experiments.append({'explore_phase_diagram': {
            'date': datetime.now(),
            'prediction_methods': prediction_methods,
            'derivative_in_y': derivative_in_y,
            'derivatives_in_x': derivatives_in_x,
            'poly_degree': poly_degree}
        })

        subfolders = ['phase_diagram']

        # ----------------------------------------
        prediction_methods = list(sorted(prediction_methods))
        if prediction_methods2plot is None:
            prediction_methods2plot = prediction_methods

        data_manager = self.get_data_manager()
        data_manager.set_X_operator(getXfunc(derivatives_in_x, poly_degree, rational))
        data_manager.set_y_operator(get_y_operator_func(self.get_derivative_in_y(derivative_in_y, derivatives_in_x)))

        # ---------- fit eqdiff ----------
        pde_finder = self.fit_eqdifff(data_manager)

        base_name = 'dery{}_derx{}_poly{}'.format(derivative_in_y, derivatives_in_x, poly_degree)
        # pde_finder = self.load_fitsave_eqdifff(self, data_manager)
        with savefig('coeficients_{}_dery{}_derx{}_poly{}'.format('_'.join(prediction_methods), derivative_in_y,
                                                                  derivatives_in_x, poly_degree), self.experiment_name,
                     subfolders=subfolders):
            self.plot_coefficients(pde_finder)

        # ---------- predictions ----------
        predictions = load_csv('phase_diagram_predictions_data', self.experiment_name, subfolders=subfolders)
        if predictions is not None and not reload and predictions.method.isin(prediction_methods).any():
            predictions = predictions.loc[~predictions.method.isin(prediction_methods), :]

        for i, prediction_method in enumerate(prediction_methods):
            if not reload or (reload and prediction_method not in predictions.method.unique()):
                df_predictions = pde_finder.integrate2(dm=data_manager,
                                                       dery=derivatives_in_x - derivative_in_y if derivative_in_y < 0 else derivative_in_y,
                                                       starting_point=starting_point,
                                                       horizon=self.phase_diagram_horizon,
                                                       method=prediction_method)
                df_predictions['method'] = prediction_method
                predictions = pd.concat([predictions if predictions is not None else
                                         pd.DataFrame([], columns=df_predictions.columns)] + [df_predictions])

        real = evaluator.get_real_values([Identity()],
                                         dm=data_manager,
                                         starting_point=starting_point,
                                         domain_variable2predict='t',
                                         horizon=self.phase_diagram_horizon)
        real = pd.concat(real)
        real = real.reset_index()
        real['method'] = 'real'

        print(predictions)
        save_csv(real, 'phase_diagram_real_data', self.experiment_name, subfolders=subfolders)
        save_csv(predictions, 'phase_diagram_predictions_data', self.experiment_name, subfolders=subfolders)

        # if we want to append new methods.
        prediction_methods = set(predictions.method.unique()).intersection(set(prediction_methods2plot))
        method_colors = {pred_method: self.colors[i] for i, pred_method in enumerate(prediction_methods)}

        # ---------- evaluate statistics ----------
        for var in data_manager.field.data:
            var_name = var.get_full_name()

            # ---------- plot phase diagram ----------
            with savefig('phase_diagram_{}_{}'.format(var_name, '-'.join(prediction_methods)), self.experiment_name,
                         subfolders=subfolders):
                fig, allax = plt.subplots(nrows=len(prediction_methods),
                                          figsize=(15, len(prediction_methods) * 15), sharex=True)
                for i, (method, df) in enumerate(
                        predictions.loc[predictions.method.isin(prediction_methods), :].groupby('method')):
                    ax = allax if len(prediction_methods) == 1 else allax[i]
                    x, dx = self.plot_phase_diagram(real[var_name].values.ravel(),
                                                    dx=data_manager.domain.step_width['t'],
                                                    method='real', var_name=var_name, color='black', ax=ax)
                    ax.set_xlim((np.min(x) - (np.max(x) - np.min(x)) / 2, np.max(x) + (np.max(x) - np.min(x)) / 2))
                    ax.set_ylim(
                        (np.min(dx) - (np.max(dx) - np.min(dx)) / 2, np.max(dx) + (np.max(dx) - np.min(dx)) / 2))
                    self.plot_phase_diagram(df[var_name].values.ravel(), dx=data_manager.domain.step_width['t'],
                                            method=method, var_name=var_name,
                                            color=method_colors[method], ax=ax)
                    ax.legend()
            plt.close("all")

            # ---------- plot series ----------
            with savefig('predictions_{}_{}'.format(var_name, '-'.join(prediction_methods)), self.experiment_name,
                         subfolders=subfolders):
                fig, allax = plt.subplots(nrows=len(prediction_methods),
                                          figsize=(15, len(prediction_methods) * 15), sharex=True)
                for i, (method, df) in enumerate(
                        predictions.loc[predictions.method.isin(prediction_methods), :].groupby('method')):
                    ax = allax if len(prediction_methods) == 1 else allax[i]
                    real_series = real[var_name].values.ravel()
                    ax.plot(real['index'].values.ravel() * data_manager.domain.step_width['t'],
                            real[var_name].values.ravel(), label='real', c='black')
                    ax.set_ylim((np.min(real_series) - (np.max(real_series) - np.min(real_series)) / 2,
                                 np.max(real_series) + (np.max(real_series) - np.min(real_series)) / 2))
                    ax.plot(real['index'].values.ravel() * data_manager.domain.step_width['t'],
                            df[var_name].values.ravel(), label='model',
                            c=method_colors[method])

                    ax.set_xlabel(data_manager.domain.axis_names[0], fontsize=20)
                    ax.set_ylabel(varname2latex(var_name, derivative=0), fontsize=20, rotation=0)
                    ax.legend()
            plt.close("all")
