#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pickle
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import os
import time
import copy

# from collections import ChainMap
from itertools import product

from sklearn.preprocessing import StandardScaler

src_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(src_path)

from src.lib.pdefind import PDEFinder, DataManager
from src.lib.variables import Variable, Domain
from src.lib.operators import PolyD, D, Poly
from src.lib.evaluators import evaluate_predictions, oracle, evaluate_integrator
import src.lib.evaluators as evaluator
import src.lib.visualization as viz

from src.scripts.config import get_filename


def varname2latex(var_name, derivative=0):
    new_var_name = var_name
    if derivative > 0:
        if derivative == 1:
            new_var_name = '\\frac{\partial ' + new_var_name + '}{\partial t}'
        else:
            new_var_name = '\\frac{\partial^' + str(derivative) + new_var_name + '}{\partial t^' + str(derivative) + '}'
    return r'{}'.format('$' + new_var_name + '$')


def convert_type(series):
    try:
        return series.astype(float)
    except:
        pass
    try:
        return series.astype(str)
    except:
        pass


# figure saver
@contextmanager
def savefig(figname, experiment, subfolders=[], verbose=False, format='eps'):
    yield
    filename = get_filename(figname, experiment, subfolders)
    if verbose:
        print('Saving in: {}'.format(filename))
    plt.savefig(filename, dpi=500, format=format)
    plt.close()


def save_csv(data, filename, experiment, subfolders=[], verbose=False):
    fname = get_filename(filename, experiment, subfolders)+'.csv'
    if verbose:
        print('Saving in: {}'.format(fname))
    data.to_csv(fname)


def load_csv(filename, experiment, subfolders=[], astypes={}, verbose=False):
    fname = get_filename(filename, experiment, subfolders)+'.csv'
    if verbose:
        print('Loading in: {}'.format(fname))
    if os.path.exists(fname):
        return pd.read_csv(fname, index_col=0).apply(convert_type)
    return None


# ---------- do/load predictions ----------
def load(path, experiment, subfolders=[]):
    filename = get_filename(path, experiment, subfolders)
    if os.path.exists(filename):
        print('loading ', filename)
        if filename.split['.'][-1] == 'csv':
            data = pd.read_csv(filename)
        elif filename.split['.'][-1] == 'pickle':
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            raise Exception(filename, 'is not pickle or csv')
        return data
    else:
        return False


def save(data, path, experiment, subfolders=[]):
    yield
    filename = get_filename(path, experiment, subfolders)
    print('Saving in: {}'.format(filename))

    if filename.split['.'][-1] == 'csv':
        data.to_csv(filename)
    elif filename.split['.'][-1] == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


# ==================== Regressors ====================
class RegressorBuilders:
    def __init__(self, name, domain_axes_name):
        self.name = str(name)
        self.domain_axes_name = domain_axes_name
        self.fit_coefs = None
        self.period = None
        self.bounds = None
        self.p0 = None
        self.fit_coefs = None

    @staticmethod
    def reg_func(*args):
        pass

    def fit(self, domain_train, values_train):
        values = values_train.data.mean(axis=tuple([ix for axname, ix in values_train.domain2axis.items()
                                                    if axname != self.domain_axes_name]))
        popt, pcov = curve_fit(self.reg_func,
                               domain_train.get_range(self.domain_axes_name)[self.domain_axes_name],
                               StandardScaler().fit_transform(values.reshape((-1, 1))).ravel(),
                               bounds=self.bounds,
                               p0=self.p0)
        self.fit_coefs = popt

    def transform(self, t):
        return self.reg_func(t, *self.fit_coefs)


class Trigonometric(RegressorBuilders):
    def __init__(self, period, domain_axes_name, units):
        RegressorBuilders.__init__(self, 'Trigonometric_peirod{}{}'.format(period, units), domain_axes_name)
        self.period = period
        self.bounds = ([self.period*0.9, -np.pi/2], [self.period*1.1, np.pi/2])
        self.p0 = [self.period, 0]

    @staticmethod
    def reg_func(t, T, fi):
        return np.cos(2 * np.pi * t / T + fi)



