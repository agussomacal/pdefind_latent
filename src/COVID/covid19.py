#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from collections import namedtuple

import matplotlib
from covid19_lib import run_model

matplotlib.use('Agg')

from src.lib.operators import DataSplit, PolyD, D, Poly

Period = namedtuple('Period', ['label', 'fecha'])
english2spanish_dict = {'[Deaths(t)]': 'Muertes', '[Cases(t)]': 'Casos',
                        'Italy': 'Italia', 'China': 'China', 'France': 'Francia', 'Spain': 'España',
                        'Italia simulada': 'Italia simulada'}

# periods = {
#     'Italy': [Period(label='Sin medidas', fecha='9/3/2020'), Period(label='Medidas', fecha='')],
#     'Spain': [Period(label='Sin medidas', fecha='14/3/2020'), Period(label='Medidas', fecha='')],
#     'France': [Period(label='Sin medidas', fecha='16/3/2020'), Period(label='Medidas', fecha='')],
#     'China': [Period(label='Sin medidas', fecha='10/2/2020'), Period(label='Medidas', fecha='')],
#     'Italia simulada': [Period(label='Sin medidas', fecha='10/3/2020'), Period(label='Medidas', fecha='')],
# }

periods = {'Italy': [Period(label='Sin medidas', fecha='8/3/2020'), Period(label='Medidas', fecha='')],
           'Spain': [Period(label='Sin medidas', fecha='14/3/2020'), Period(label='Medidas', fecha='')],
           'France': [Period(label='Sin medidas', fecha='16/3/2020'), Period(label='Medidas', fecha='')],
           'China': [Period(label='Sin medidas', fecha='10/2/2020'), Period(label='Medidas', fecha='')]}#20/2/2020


periods = {'Italy': [Period(label='Periodo completo', fecha='')],
           'Spain': [Period(label='Periodo completo', fecha='')],
           'France': [Period(label='Periodo completo', fecha='')],
           'China': [Period(label='Periodo completo', fecha='')]}#20/2/2020

# ========================================================= #
# ---------------------- parameters ----------------------- #
experiment_name = 'COVID19'
type_of_experiment = 'Casos'  # name to save the files if we change specifications.
filename = "COVID-19-geographic-disbtribution-worldwide-2020-03-24.xlsx"  # loading data
# filename = "italia-simulada2.csv"

accepted_variables = ['Todas']  # ['Muertes', 'Casos', 'Recuperados', 'Todas']
countries = ['China', 'Italia']
# countries = ['Italia simulada']
# countries = ['France', 'Spain']
# countries = ['Italy']
# countries = ['Italy']

use_lasso = True  # use lasso to fit a general model. Otherwise is a linear regression over the specific model that
# should be specified in operator_x and operator_y

# ------- specifing general model -------
target_derivative_order = 2  # diferential equation of order ...
polynomial_order = 3  # order of polynomial combinations:
# ej: poly=2 and targetder=2-> dictionary = [1, F, F', F^2, FF', F'^2] try to fit -> [F'']

do_cumulative = True  # depende como venga la data, si es la suma de muertes hasta el momento => do_cumulative=False
# porque ya viene acumulada. Sino True. Para COVID-19-geographic-disbtribution-worldwide-2020-03-22.xlsx hay que usar
# True porque no viene acumulada, son los casos por día.
death_threshold_2_begin = 0  # para sacarse de encima los días iniciales con 0 muertes/casos donde la epidemia
# aún no comenzó. A partir de qque encuentra un día con ese valor a partir de ahi toma los datos.

# sometimes there is too few data after measures then, a hardcoded shift to gather some days (ej 5) before the measure
# can gave better results... At a cost oviously.
days_before_meassure_to_gather_data = 0

# ------ train and test sets ------
ptrain = 1  # p=1 means all data is used for training. If p=0.7 the first 70% is used
init_condition = 0.7
ptest = 1-init_condition  # p=0.7 means the last 70% of the data is used for testing and the initial condition for predictions is
# taken then in exactly the 30% (because there is 70% at the right)

# when predicting, predict all the test set (=1) and go to the future (>1). If 10 days are in the testset,
# then =1.5 means 15 days of predictions
prediction_horizon_proportion = {'Sin medidas': 1.5, 'Medidas': 1.5, 'Periodo completo': 1.3}

if use_lasso:
    type_of_experiment = type_of_experiment + '_lasso'
    with_mean = True
    with_std = True

    # defines the operator that generates the dictionary of functions that will be use to fit the target y operator.
    def x_operator_func(rational=False):
        def x_operator(field, regressors):
            'field = [M, C]'
            new_field = copy.deepcopy(field)
            new_field = PolyD({'t': target_derivative_order - 1}) * new_field
            "[M, C, M', C', M'', C'' ...]"
            if rational:
                new_field.append(new_field.__rtruediv__(1.0))
            new_field = Poly(polynomial_order=polynomial_order) * new_field
            "[1, M, C, M', C', M'', C'' ..., MC, MM'', MCM'...]"
            return new_field

        return x_operator

    # defines the operator that generates the target y operator.
    def y_operator_func():
        def y_operator(field):
            return D(derivative_order=target_derivative_order, axis_name='t') * field

        return y_operator

else:
    type_of_experiment = type_of_experiment + '_linreg'
    with_mean = False
    with_std = False

    def x_operator_func(rational=False):
        def x_operator(field, regressors):
            new_field = D(1, 't') * field  # M'
            new_field.append(((D(1, 't') * field) ** 2))   # (M')^2
            new_field.append((field * (D(1, 't') * field)))  # MM'
            return new_field

        # equación diferencial
        # M'' = a*M' + b*M'^2 + c*MM'
        # y = <c_vect, X_vect>
        # Xvect = (M', M'^2, MM')
        # c_vect = (a, b, c)

        return x_operator

    # defines the operator that generates the target y operator.
    def y_operator_func():
        def y_operator(field):
            return D(derivative_order=target_derivative_order, axis_name='t') * field

        return y_operator

run_model(filename=filename,
          experiment_name=experiment_name,
          type_of_experiment=type_of_experiment,
          countries=countries,
          periods=periods,
          death_threshold_2_begin=death_threshold_2_begin,
          cumulative=do_cumulative,
          days_before_meassure_to_gather_data=days_before_meassure_to_gather_data,
          prediction_horizon_proportion=prediction_horizon_proportion,
          with_mean=with_mean,
          with_std=with_std,
          use_lasso=use_lasso,
          trainSplit=DataSplit(axis_percentage_dict={"t": ptrain}),
          testSplit=DataSplit(axis_percentage_dict={"t": ptest}, axis_init_percentage_dict={"t": 1 - ptest}),
          x_operator_func=x_operator_func,
          y_operator_func=y_operator_func,
          accepted_variables=accepted_variables)
