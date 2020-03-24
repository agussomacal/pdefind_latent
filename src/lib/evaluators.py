import time
from contextlib import contextmanager

import pandas as pd
import numpy as np

from src.lib.operators import DataSplitOnIndex
from src.lib.simulation_manager import Integrator


@contextmanager
def timeit(msg):
    t0 = time.time()
    yield
    print('Duracion {}: {}'.format(msg, time.time()-t0))


def get_real_values(split_data_operator_list, dm, starting_point, domain_variable2predict, horizon):
    """

    :param split_data_operator_list:
    :param dm: DataManager
    :param starting_point: when more than one variable it is used to define the other domain points.
    :param domain_variable2predict:
    :param horizon:
    :return:
    """

    var_names = [var.get_full_name() for var in dm.field.data]
    real_values = []
    for i, split_operation in enumerate(split_data_operator_list):
        temp_field = split_operation * dm.field

        value_dict = starting_point.copy()
        df_real_values = pd.DataFrame(np.nan, columns=var_names, index=list(range(horizon)))
        for h in range(horizon):
            value_dict[domain_variable2predict] = h - horizon
            df_real_values.loc[h, :] = pd.Series(temp_field.evaluate_ix(value_dict))
        df_real_values['random_split'] = i
        df_real_values = df_real_values.astype(float)
        real_values.append(df_real_values)
    return real_values


def evaluate_integrator(pdfinder, data_split_operator, dm, starting_point,
                        domain_variable2predict, horizon, num_evaluations, method):
    num_points = dm.field.domain.shape[domain_variable2predict]
    num_points *= data_split_operator.axis_percentage_dict[domain_variable2predict]

    # assumes lags are smaller than horizons.
    spliting_points_predict = np.arange(horizon, num_points - horizon, horizon, dtype=int)
    # np.arange(2 * horizon - 2, num_points, horizon, dtype=int)
    spliting_points_real = spliting_points_predict + horizon - 1
    max_splits = min(len(spliting_points_real), len(spliting_points_predict))
    which_splittings = np.random.choice(max_splits, size=min(max_splits, num_evaluations), replace=False)
    # which_splittings = list(range(min(len(spliting_points_predict), num_evaluations)))

    predictions_df_list = pdfinder.integrate([DataSplitOnIndex({domain_variable2predict: spp}) * data_split_operator
                                            for spp in spliting_points_predict[which_splittings]],
                                            dm,
                                            starting_point=starting_point,
                                            domain_variable2predict=domain_variable2predict,
                                            horizon=horizon,
                                            method=method)

    real_df_list = get_real_values([DataSplitOnIndex({domain_variable2predict: spr}) * data_split_operator
                                    for spr in spliting_points_real[which_splittings]],
                                   dm,
                                   starting_point,
                                   domain_variable2predict,
                                   horizon)

    true = {}
    pred = {}
    for var in dm.field.data:
        var_name = var.get_full_name()
        true[var_name] = pd.DataFrame(np.nan, columns=list(range(horizon)), index=list(range(num_evaluations)))
        pred[var_name] = pd.DataFrame(np.nan, columns=list(range(horizon)), index=list(range(num_evaluations)))
        for i, (r, p) in enumerate(zip(real_df_list, predictions_df_list)):
            true[var_name].iloc[i, :] = r[var_name]
            pred[var_name].iloc[i, :] = p[var_name]

    return true, pred


def evaluate_predictions(pdfinder, data_split_operator, dm, starting_point, domain_variable2predict, horizon,
                         num_evaluations, dery, prediction_methods=('algebraic',), seed=0):
    np.random.seed(seed)

    num_points = dm.field.domain.shape[domain_variable2predict]
    num_points *= data_split_operator.axis_percentage_dict[domain_variable2predict]

    # assumes lags are smaller than horizons.
    spliting_points_predict = np.arange(horizon, num_points - horizon, horizon, dtype=int)
    spliting_points_real = spliting_points_predict + horizon - 1 # np.arange(2 * horizon - 2, num_points, horizon, dtype=int)
    max_splits = min(len(spliting_points_real), len(spliting_points_predict))
    which_splittings = np.random.choice(max_splits, size=min(max_splits, num_evaluations), replace=False)
    # which_splittings = list(range(min(len(spliting_points_predict), num_evaluations)))

    # ---------- real data points ----------
    with timeit('real points'):
        real = get_real_values([DataSplitOnIndex({domain_variable2predict: spr}) * data_split_operator
                                for spr in spliting_points_real[which_splittings]],
                               dm,
                               starting_point,
                               domain_variable2predict,
                               horizon)
        real = pd.concat(real)
        real = real.reset_index()
        real['method'] = 'real'

    # ---------- predictions ----------
    predictions = pd.DataFrame()
    for method in prediction_methods:
        print('\t *method {}'.format(method))
        with timeit(method):
            if 'algebraic' in method:
                # ---------- our prediction using sympy ----------
                predictions_temp = pdfinder.predict(
                    [DataSplitOnIndex({domain_variable2predict: spp}) * data_split_operator
                     for spp in spliting_points_predict[which_splittings]],
                    dm,
                    starting_point=starting_point,
                    domain_variable2predict=domain_variable2predict,
                    horizon=horizon)
            else:
                # ---------- predictions using integrators ----------
                if 'RK4' in method:
                    method_temp = 'RK4'
                elif 'AdamsBashMoulton3' in method:
                    method_temp = 'AdamsBashMoulton3'
                elif 'Euler' in method:
                    method_temp = 'Euler'
                else:
                    method_temp = method
                predictions_temp = pdfinder.integrate2(dm, dery=dery,
                                                       starting_point=starting_point,
                                                       horizon=horizon,
                                                       method=method_temp)
                # predictions_temp = pdfinder.integrate(
                #     [DataSplitOnIndex({domain_variable2predict: spp}) * data_split_operator
                #      for spp in spliting_points_predict[which_splittings]],
                #     dm,
                #     starting_point=starting_point,
                #     domain_variable2predict=domain_variable2predict,
                #     horizon=horizon,
                #     method=method_temp)
        # predictions_temp = pd.concat(predictions_temp)
        predictions_temp = predictions_temp.reset_index()
        predictions_temp['method'] = method
        predictions = predictions.append(predictions_temp)

    if len(prediction_methods) > 0:
        predictions = predictions.set_index('index')

    # predictions[['index', 'Theta(t)']] = predictions[['index', 'Theta(t)']].astype(float)
    return real.set_index('index'), predictions

#
# def ExpSmoothing(train, test):
#     fit = ExponentialSmoothing(saledata, seasonal_periods=4, trend='add', seasonal='add', damped=True).fit()
#     fcast = fit.forecast(12).rename(r'$\alpha=%s$' % fit3.model.params['smoothing_level'])
#     # plot
#     fcast3.plot(marker='o', color='green', legend=True)
#     fit3.fittedvalues.plot(marker='o', color='green')
#
#     plt.show()


def oracle(dict_pred, eq_diff_model, integration_dt):
    """
    Uses the true differential equation to predict the next horizon point given the predictions.
    :return:
    """
    var_names = [var_name for var_name in dict_pred.keys()]
    horizons = min([df.shape[1] for df in dict_pred.values()])
    num_trys = min([df.shape[0] for df in dict_pred.values()])
    next_step_pred = {var_name: pd.DataFrame(np.nan,
                                             columns=list(range(1, horizons)),
                                             index=list(range(num_trys)))
                      for var_name in var_names}

    for h in range(0, horizons-1):
        for t in range(num_trys):
            p_init = [dict_pred[var_name].loc[t, h] for var_name in eq_diff_model.var_names]
            new_pred = Integrator.integrate(model=eq_diff_model,
                                            Xinit=np.array(p_init),
                                            time_steps=1,
                                            integration_dt=integration_dt)
            for p, var_name in zip(new_pred.T, eq_diff_model.var_names):
                next_step_pred[var_name].loc[t, h+1] = p

    last_horizon_pred = {var_name: pd.DataFrame(np.nan,
                                                columns=list(range(1)),
                                                index=list(range(num_trys)))
                         for var_name in var_names}
    for t in range(num_trys):
        p_init = [dict_pred[var_name].loc[t, 1] for var_name in eq_diff_model.var_names]
        new_pred = Integrator.integrate(model=eq_diff_model,
                                        Xinit=np.array(p_init),
                                        time_steps=horizons-1,
                                        integration_dt=integration_dt)
        for p, var_name in zip(new_pred.T, eq_diff_model.var_names):
            last_horizon_pred[var_name].loc[t, 0] = p[-1]

    return next_step_pred, last_horizon_pred


def error(yhat, y, axis=0):
    return np.sqrt(((yhat - y) ** 2).sum(axis))


def rsquare(yhat, y, axis=0):
    try:
        return 1 - ((yhat - y) ** 2).sum(axis) / ((y - y.mean(axis)) ** 2).sum(axis)
    except:
        return np.nan


def mape(yhat, y, axis=0):
    return np.abs((y - yhat) / y).mean(axis)


def mape_sd(yhat, y, axis=0):
    return np.abs((y - yhat) / y).std(axis)


def apply_to_dict(yhat_dict, y_dict, func, indexes=slice(None)):
    return {var_name: func(yhat_dict[var_name].values, y_dict[var_name].iloc[:, indexes].values) for var_name in y_dict.keys()}
