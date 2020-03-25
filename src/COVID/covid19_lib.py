#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import os
from collections import namedtuple

import matplotlib
from lib.evaluators import timeit
from lib.pdefind import DataManager, PDEFinder
from tqdm import tqdm

matplotlib.use('Agg')

import matplotlib.pylab as plt
import pandas as pd
import numpy as np

import src.scripts.config as config
from src.lib.variables import Variable, Domain, Field

from src.lib.experiments import ExperimentSetting
from src.scripts.utils import savefig

Period = namedtuple('Period', ['label', 'fecha'])
english2spanish_dict = {'[Deaths(t)]': 'Muertes', '[Cases(t)]': 'Casos',
                        'Italy': 'Italia', 'China': 'China', 'France': 'Francia', 'Spain': 'España',
                        'Italia simulada': 'Italia simulada'}


spanish2english_dict = {'Muertes': '[Deaths(t)]', 'Casos': '[Cases(t)]', 'Recuperados': '[Healed(t)]'}


def mse(x, y):
    return np.sum((np.asarray(x).ravel() - np.asarray(y).ravel()) ** 2)


class CovidExperimentSetting(ExperimentSetting):
    def __init__(self, filename, experiment_name, type_of_experiment, countries, prediction_horizon_proportion,
                 death_threshold_2_begin=0, recalculate=False, cumulative=True, periods={},
                 days_before_meassure_to_gather_data=0, accepted_variables=['[Deaths(t)]']):
        ExperimentSetting.__init__(self, experiment_name=experiment_name)

        self.prediction_horizon_proportion = prediction_horizon_proportion
        self.type_of_experiment = type_of_experiment
        self.days_before_meassure_to_gather_data = days_before_meassure_to_gather_data
        self.filename = filename
        self.recalculate = recalculate
        self.stats = pd.DataFrame([])
        self.periods = periods
        self.data = {}
        self.df_data = None
        self.t = np.array([])
        self.countries = countries
        self.cumulative = cumulative
        self.death_threshold_2_begin = death_threshold_2_begin
        self.info = {}
        self.accepted_variables = [spanish2english_dict[av] for av in accepted_variables]

    def get_country_data(self):
        data_filename = '{}/data_covid19/{}'.format(
            config.project_dir, self.filename)
        if 'xlsx' in data_filename:
            data = pd.read_excel(data_filename)
        elif "italia-simulada2.csv" in data_filename:
            new_data = pd.read_csv(data_filename, names=['DateRep', 'Deaths'])
            new_data = new_data['Deaths'].values
            intermediate_day_m = (new_data[1:] + new_data[:-1]) / 2
            data = []
            for idm, m in zip(intermediate_day_m, new_data):
                data += [m, idm]  # + int(np.random.normal(idm, scale=(idm-m)/2))
            data = pd.DataFrame(data, columns=['Deaths'])
            data['DateRep'] = pd.date_range(pd.to_datetime('22/3/2020', dayfirst=True), freq='D',
                                            periods=data.shape[0]) - data.shape[0] + 1

            data['Cases'] = data['Deaths']

            # data['DateRep'] = pd.date_range(pd.to_datetime('22/3/2020', dayfirst=True), freq='2D', periods=data.shape[0])-data.shape[0]+1
            data['Countries and territories'] = 'Italia simulada'
        else:
            raise Exception('No data for that name')

        data.DateRep = pd.to_datetime(data.DateRep)
        self.df_data = data

        if 'all' in self.countries:

            countries = data['Countries and territories'].unique().tolist() + [c for c in self.countries if c != 'all']
            countries = set(countries)
        else:
            countries = self.countries

        for country in countries:
            if country in data['Countries and territories'].values:
                country_data = data.loc[data['Countries and territories'] == country, :]
            elif country.lower() == 'world':
                country_data = data.groupby('DateRep').sum()
                country_data.reset_index(inplace=True)
            elif country == 'test_exponential':
                t = np.linspace(0, 1, 100)
                C = np.diff(np.exp(2 * t))
                D = np.diff(np.exp(0.2 * t))
                country_data = pd.DataFrame(np.array([t[1:], C, D]).T, columns=['DateRep', 'Cases', 'Deaths'])
            else:
                continue
            country_data = country_data.sort_values(by='DateRep')
            if self.cumulative:
                country_data['Deaths'] = country_data['Deaths'].cumsum()
                country_data['Cases'] = country_data['Cases'].cumsum()

            begin_ix = np.where(country_data['Deaths'] >= self.death_threshold_2_begin)[0]
            if len(begin_ix) == 0:
                continue
            else:
                begin_ix = begin_ix.min()
            country_data = country_data.iloc[begin_ix:, :]
            begin_ix = 0

            country_data.reset_index(inplace=True)
            country_data['total_days'] = country_data.index

            if country not in self.periods.keys():
                self.periods[country] = [
                    Period(label='No hay informacion sobre medidas', fecha=country_data['DateRep'].values[-1])]

            for p in self.periods[country]:
                end_ix = np.array(np.where(country_data['DateRep'] == pd.to_datetime(p.fecha, dayfirst=True))).ravel()
                end_ix = -1 if len(end_ix) == 0 else end_ix.min()
                yield country_data.iloc[begin_ix:end_ix, :], country, p
                begin_ix = end_ix - self.days_before_meassure_to_gather_data  # TODO: Harcoded!!

    def set_underlying_model(self, df):
        """
        # ---------- simulating data ----------
        :param dt:
        :param time_steps:
        :param odespy_method:
        :return:
        """
        self.t = df['total_days'].values
        self.data['Deaths'] = df['Deaths'].values
        self.data['Cases'] = df['Cases'].values

    def get_domain(self):
        return Domain(lower_limits_dict={"t": self.t[0]},
                      upper_limits_dict={"t": self.t[-1]},
                      step_width_dict={"t": np.median(np.diff(self.t))})

    def get_variables(self):
        domain = self.get_domain()
        return [Variable(measurements, domain, domain2axis={"t": 0}, variable_name=var_name) for var_name, measurements
                in self.data.items()]

    def fit_eqdifff(self, data_manager):
        with timeit('pdefind fitting'):
            pde_finder = PDEFinder(with_mean=self.with_mean, with_std=self.with_std, use_lasso=self.use_lasso)
            pde_finder.set_fitting_parameters(cv=self.cv, n_alphas=self.alphas, max_iter=self.max_iter,
                                              alphas=self.alpha)
            X = data_manager.get_X_dframe(self.trainSplit)
            Y = data_manager.get_y_dframe(self.trainSplit)

            # filter cases where the function is abobe a threshold
            # sample = np.where((self.trainSplit * data_manager.field.data[0]).data >= self.far_from_detector_threshold)[
            #     0]
            # print('Samples above threshold of {}: {}'.format(self.far_from_detector_threshold, len(sample)))
            # X = X.iloc[sample, :]
            # Y = Y.iloc[sample, :]

            if X.shape[0] > self.max_train_cases:
                sample = np.random.choice(X.shape[0], size=self.max_train_cases)
                X = X.iloc[sample, :]
                Y = Y.iloc[sample, :]

            pde_finder.fit(X, Y)
        return pde_finder

    def explore(self, x_operator_func, y_operator_func, rational=False):
        subfolders = [self.type_of_experiment]
        for df, country, period in self.get_country_data():
            print('\n\n========== ========== ========== ==========')
            print('Exploring {}'.format(country))
            if country not in self.info.keys():
                self.info[country] = {}

            self.set_underlying_model(df)

            for variable in [variable for variable in self.get_variables()] + [self.get_variables()]:
                stats = pd.DataFrame([])
                variable = Field(variable)
                base_name = str(variable)
                if base_name not in self.accepted_variables:
                    continue
                print('\nVariable {}'.format(base_name))

                if base_name not in self.info[country].keys():
                    self.info[country][base_name] = []

                # ---------- fit eqdiff ----------
                data_manager = DataManager()
                data_manager.add_variables(variable)
                # data_manager.add_regressors(self.get_regressors())
                data_manager.set_domain()

                data_manager.set_X_operator(x_operator_func(rational=rational))
                data_manager.set_y_operator(y_operator_func())
                pde_finder = self.fit_eqdifff(data_manager)
                stats = pd.concat([stats, pd.concat([pd.DataFrame([[country, period.label, period.fecha]],
                                                                            index=pde_finder.coefs_.index,
                                                                            columns=['country', 'medidas',
                                                                                   'fecha_final']),
                                                               pde_finder.coefs_],
                                                              axis=1)], axis=0)
                # ---------- plot ----------
                with savefig('{}_{}_coeficients.png'.format(base_name, country), self.experiment_name,
                             subfolders=subfolders, format='png'):
                    self.plot_coefficients(pde_finder)
                    plt.xscale('log')

                with savefig('{}_{}_fitvsreal.png'.format(base_name, country), self.experiment_name,
                             subfolders=subfolders, format='png'):
                    self.plot_fitted_and_real(pde_finder, data_manager, col="blue", subinit=None, sublen=None)

                # --------- predictions ---------
                predictions_temp = self.optimize_predictions(pde_finder, variable, x_operator_func, y_operator_func,
                                                             data_manager, period, rational)

                self.info[country][base_name].append({'coefs': pde_finder.coefs_,
                                                      'period': period,
                                                      'data_real': data_manager.field,
                                                      'predictions': predictions_temp,
                                                      'data_raw': df})

                stats.to_csv(config.get_filename(filename='{}_coefs.csv'.format(base_name),
                                                      experiment=self.experiment_name,
                                                      subfolders=[self.type_of_experiment]))
                self.plot_results()

    def optimize_predictions(self, pde_finder, variable, x_operator_func, y_operator_func, data_manager, period,
                             rational):
        if np.sum(np.abs(pde_finder.coefs_.values)) > 0:
            data_manager_test = DataManager()
            data_manager_test.add_variables(self.testSplit * variable)
            # data_manager.add_regressors(self.get_regressors())
            data_manager_test.set_domain()

            data_manager_test.set_X_operator(x_operator_func(rational=rational))
            data_manager_test.set_y_operator(y_operator_func())

            t = np.arange(data_manager_test.domain.lower_limits['t'],
                          data_manager.domain.upper_limits['t'] * self.prediction_horizon_proportion[
                              period.label],
                          data_manager.domain.step_width['t'])

            der1 = (data_manager_test.field.data[0].data[1] - data_manager_test.field.data[0].data[0]) / \
                   data_manager_test.domain.step_width['t']
            der2 = (data_manager_test.field.data[0].data[2] - data_manager_test.field.data[0].data[0]) / \
                   data_manager_test.domain.step_width['t'] / 2

            var_name = str(data_manager_test.field.data[0])
            loss_best = np.Inf
            predictions = []
            # random around the derivative to get the initial conditions beetter matchin g with observations.
            for r in tqdm(np.random.normal(loc=(der1 + der2) / 2, scale=np.abs(der1 - der2) / 2, size=20).tolist() +
                          [der1, der2, (der1 + der2) / 2], desc='Optimizing predictions'):
                v0 = [data_manager_test.field.data[0].data[0], r]
                predictions_temp = pde_finder.integrate3(dm=data_manager_test,
                                                         t=t,
                                                         v0=v0)
                loss = mse(predictions_temp[var_name].values[np.arange(data_manager_test.field.data[0].shape[0])],
                            data_manager_test.field.data[0].data)
                if loss_best > loss:
                    predictions = predictions_temp

            return predictions

    def plot_results(self):
        for country, vars_info in self.info.items():
            for var, list_info in vars_info.items():
                original_var_name = str(list_info[0]['data_real'])
                var_name = english2spanish_dict[original_var_name]
                lines = {}
                with savefig('{}_predict_{}.png'.format(country, var_name), self.experiment_name,
                             subfolders=[self.type_of_experiment], format='png'):
                    plt.title(english2spanish_dict[country] + ' ' + var_name.lower())

                    # ------ plot real ------
                    country_data = self.df_data.loc[self.df_data['Countries and territories'] == country, :]
                    country_data = country_data.sort_values(by='DateRep')
                    if self.cumulative:
                        country_data['Deaths'] = country_data['Deaths'].cumsum()
                        country_data['Cases'] = country_data['Cases'].cumsum()

                    lab = 'Data real {}'.format(var_name.lower())
                    lines[lab], = plt.plot(country_data['DateRep'], country_data[original_var_name[1:-4]], '.-',
                                           c='tab:green', label=lab)

                    plt.xlabel('Time')
                    plt.ylabel(var_name)

                    # xmin = np.Inf
                    ymax = 0
                    for info in list_info:
                        ymax = np.max((ymax, max([d.data.max() * 2 for d in info['data_real'].data])))
                        # xmin = np.min((xmin, min([d.data.min() for d in info['data_real'].data])))

                        # plt.xlim(left=xmin)
                        plt.ylim((0, ymax))

                        real = []
                        t_real = []
                        for var in info['data_real'].data:
                            lab = 'Data real {} for train'.format(var_name.lower())
                            real += var.data.tolist()
                            dt = var.domain.step_width['t']
                            t_real += np.arange(var.domain.lower_limits['t'], var.domain.upper_limits['t'] + dt,
                                                dt).tolist()
                            t_real = info['data_raw'].loc[info['data_raw']['total_days'].isin(t_real), 'DateRep']
                            lines[lab], = plt.plot(t_real, real, '.-', c='k', label=lab)

                        for var in info['data_real'].data:
                            lab = 'Predictcion {}'.format(info['period'].label.lower())
                            tmin = info['data_raw'].loc[
                                info['data_raw']['total_days'].isin(info['predictions'].index), 'DateRep'].values.min()
                            t = pd.date_range(tmin, periods=info['predictions'].shape[0], freq='D')

                            lines[lab], = plt.plot(t,
                                                   info['predictions'].loc[:, str(var)].values, '-',
                                                   label=lab)

                            tosave = info['predictions']
                            tosave['real'] = np.nan

                            tosave.loc[[True if i in t_real.tolist() else False for i in t], 'real'] = [r for i, r in
                                                                                                        zip(t_real,
                                                                                                            real) if
                                                                                                        i in t]
                            tosave.to_csv(
                                config.get_filename(filename='predictions_{}.csv'.format(info['period'].label.lower()),
                                                    experiment=self.experiment_name,
                                                    subfolders=[self.type_of_experiment]))

                        try:
                            lab = 'Fin {}'.format(info['period'].label.lower())
                            lines[lab] = plt.axvline(pd.to_datetime(info['period'].fecha, dayfirst=True), c='r',
                                                     linestyle='-.', ymin=0,
                                                     ymax=ymax / 2,
                                                     label=lab)
                        except:
                            pass

                    plt.tight_layout()
                    plt.grid(axis='x', color='gray', linestyle='-.', linewidth=1, alpha=0.65)
                    plt.legend(list(lines.values()), list(lines.keys()))


def run_model(filename, experiment_name, type_of_experiment, countries, periods, death_threshold_2_begin, cumulative,
              days_before_meassure_to_gather_data, prediction_horizon_proportion, with_mean, with_std, use_lasso,
              trainSplit, testSplit, x_operator_func, y_operator_func, accepted_variables):
    experiment = CovidExperimentSetting(filename=filename,
                                        experiment_name=experiment_name,
                                        type_of_experiment=type_of_experiment,
                                        countries=countries,
                                        death_threshold_2_begin=death_threshold_2_begin,
                                        recalculate=False,
                                        cumulative=cumulative,
                                        periods=periods,
                                        days_before_meassure_to_gather_data=days_before_meassure_to_gather_data,
                                        prediction_horizon_proportion=prediction_horizon_proportion,
                                        accepted_variables=accepted_variables)

    experiment.set_plotting_params(sub_set_init=100, sub_set_len=100)
    experiment.set_pdefind_params(with_mean=with_mean, with_std=with_std, alphas=100, max_iter=10000, cv=5,
                                  max_train_cases=50000, use_lasso=use_lasso)
    experiment.set_data_split(trainSplit=trainSplit, testSplit=testSplit)

    experiment.explore(x_operator_func=x_operator_func,
                       y_operator_func=y_operator_func,
                       rational=False)
