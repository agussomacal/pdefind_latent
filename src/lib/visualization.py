import matplotlib.pylab as plt
from src.lib.variables import Field
from src.lib.evaluators import rsquare
import pandas as pd
import numpy as np


def plot_real_vs_fitted(y, yhat, col):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    plt.rcParams.update({'axes.labelsize': 'x-large'})
    ax[0].plot(y, yhat, '.', c=col, label=r'$\frac{\partial f}{\partial t}$', alpha=0.5)
    ax[0].set_ylabel(r'$\frac{\partial \hat f}{\partial t}$')
    ax[0].legend()

    ax[1].plot(y, yhat - y, '.', c="black", label='residual', alpha=0.2)
    ax[1].set_xlabel(r'$\frac{\partial f}{\partial t}$')
    ax[1].set_ylabel(r'$\frac{\partial \hat f}{\partial t} - \frac{\partial f}{\partial t}$', fontsize=15)
    ax[1].legend()


def plot_func_and_residuals_in_time(t, y, yhat, col):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    plt.rcParams.update({'axes.labelsize': 'x-large'})
    ax[0].plot(t, y, '-', c="black", label=r'$\frac{\partial f}{\partial t}$', alpha=0.7)
    ax[0].plot(t, yhat, '-', c=col, label=r'$\frac{\partial f}{\partial t}$', alpha=0.7)
    ax[0].set_ylabel(r'$\frac{\partial }{\partial t}$', fontsize=15)
    ax[0].legend()

    ax[1].plot(t, yhat - y, '-', c="black", label='residual', alpha=0.7)
    ax[1].set_xlabel('t', fontsize=15)
    ax[1].set_ylabel(r'$\frac{\partial \hat f}{\partial t} - \frac{\partial f}{\partial t}$', fontsize=15)
    ax[1].legend()


def plot_mape(var_name, col, path, mape, mape_next_point=None, mape_last_horizon=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    ax.plot(mape[var_name], '.-', c=col, label=var_name)
    if mape_next_point is not None:
        ax.plot(np.arange(1, len(mape_next_point[var_name])+1), mape_next_point[var_name].ravel(), '.-', c="gray",
                label="perfect prediction next point")
    if mape_last_horizon is not None:
        ax.hlines(mape_last_horizon[var_name], color="black", label="perfect prediction after first prediction", xmin=0,
                  xmax=len(mape[var_name].ravel())-1)

    ax.set_ylabel("mape")
    ax.legend()
    ax.set_title('mape' + ' for ' + var_name)
    ax.set_xlabel("Horizon future steps")

    fig.suptitle("Mape for prediction for {}".format(var_name))
    plt.savefig(path + "_mape_predictions_{}.png".format(var_name))


def plot_rsquare(var_name, col, path, rsq, rsq_last_horizon=None, rsq_next_point=None, ax=None, fig=None):
    rsq = rsq[var_name][rsq[var_name] >= 0]

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    ax.plot(rsq, '.-', c=col, label=var_name)
    if rsq_next_point is not None:
        rsq_next_point = rsq_next_point[var_name][rsq_next_point[var_name] >= 0]
        ax.plot(np.arange(1, len(rsq_next_point)+1), rsq_next_point.ravel(), '.-', c="gray",
                label="perfect prediction next point")

    if rsq_last_horizon is not None:
        rsq_last_horizon = rsq_last_horizon[var_name][rsq_last_horizon[var_name] >= 0]
        ax.hlines(rsq_last_horizon, color="black", label="perfect prediction after first prediction", xmin=0,
                  xmax=len(rsq.ravel())-1)

    ax.set_ylabel(r'$R^2$')
    ax.legend()
    ax.set_title(r'$R^2$' + ' for ' + var_name)
    ax.set_xlabel("Horizon future steps")

    fig.suptitle("R for prediction for {}".format(var_name))
    plt.savefig(path + "_rsq_predictions_{}.png".format(var_name))


def sympy_derivative_name_to_proper_name(name):
    split = name.split("Derivative(")[1].split(",")
    var_name = split[0].split("(")[0]
    if len(split) == 2:
        der = split[1].split(")")[0].strip()
    else:
        return name
        # der = split[1].split("(")[1]+"^"+split[2].split(")")[0].strip()

    return "d{}/d{}".format(var_name, der)


def plot_coefficients(coefs_, path):
    temp_coefs = coefs_.T
    temp_coefs.sort_index(inplace=True)
    temp_coefs["len"] = pd.Series(temp_coefs.index).apply(len).values
    temp_coefs.sort_values(by=["len"], inplace=True)
    temp_coefs.drop("len", inplace=True, axis=1)
    temp_coefs.index = [1 if c == "1.00000000000000" else c for c in temp_coefs.index]
    temp_coefs.columns = [sympy_derivative_name_to_proper_name(c) for c in temp_coefs.columns]
    temp_coefs.plot(y=temp_coefs.columns, kind="barh")
    plt.title("Fitted coefficients")
    plt.xlabel("Coefficient values")
    plt.tight_layout()
    plt.savefig(path + "Coefficients.png")

