import numpy as np
import pandas as pd
from scipy.integrate import odeint
import odespy


class DifferentialModels:
    def __init__(self, var_names):
        self.var_names = var_names

    def odespy_func(self):
        def f(u, t):
            return self.get_dt(u)
        return f

    def get_dt(self, X):
        return X


# ============================================================= #
#                       Lorenz attractor
# ============================================================= #
class LorenzAttractor(DifferentialModels):
    def __init__(self, sigma=10, rho=28, beta=8.0 / 3):
        DifferentialModels.__init__(self, var_names=["X(t)", "Y(t)", "Z(t)"])
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def get_dt(self, X):
        return np.array((self.sigma * (X[1] - X[0]),
                         X[0] * (self.rho - X[2]) - X[1],
                         X[0] * X[1] - self.beta * X[2]))


# ============================================================= #
#                       Roseler attractor
# ============================================================= #
class RoselerAttractor(DifferentialModels):
    def __init__(self, a=0.52, b=2.0, c=4.0):
        DifferentialModels.__init__(self, var_names=["X(t)", "Y(t)", "Z(t)"])
        self.a = a
        self.b = b
        self.c = c

    def get_dt(self, X):
        return np.array((- X[1] - X[2],
                         X[0] + self.a * X[1],
                         self.b + X[2] * (X[0] - self.c)))


# ============================================================= #
#                       Van der Pol attractor
# ============================================================= #
class VanDerPolAttractor(DifferentialModels):
    def __init__(self, mu=0.01):
        DifferentialModels.__init__(self, var_names=["X(t)", "Y(t)"])
        self.mu = mu

    def get_dt(self, X):
        return np.array((
            X[1],
            self.mu*(1-X[0]**2)*X[1]-X[0]
        ))


# ============================================================= #
#                       Lorenz attractor X
# ============================================================= #
class LorenzXAttractor(DifferentialModels):
    def __init__(self, sigma=10, rho=28, beta=8.0 / 3):
        DifferentialModels.__init__(self, var_names=["X(t)", "dX(t)"])
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def get_dt(self, X):
        return np.array((
            X[1],
            self.sigma*X[0]/2*(self.rho+self.sigma*self.rho-(1+self.sigma))-X[1]/2*(self.sigma+1)**2
        ))


# ============================================================= #
#                       Pendulus
# ============================================================= #
class Pendulus(DifferentialModels):
    def __init__(self, c=10):
        DifferentialModels.__init__(self, var_names=["Omega", "Theta"])
        self.c = c

    def get_dt(self, X):
        theta, omega = X
        return [omega, -self.c * np.sin(theta)]


# ============================================================= #
#                       Oscillator
# ============================================================= #
class oscilator(DifferentialModels):
    def __init__(self, A):
        DifferentialModels.__init__(self, var_names=["X", "Y"])
        self.A = A

    def get_dt(self, X):
        return np.matmul(self.A, X)


# ============================================================= #
#                       Oscillator
# ============================================================= #
class StressedString(DifferentialModels):
    def __init__(self, L, k, m, A, g=10):
        DifferentialModels.__init__(self, var_names=['u'])
        self.L = L
        self.k = k
        self.m = m
        self.A = A
        self.g = g

    def get_dt(self, X):
        return


# ============================================================= #
#            Eq_diff Integrator for simulations
# ============================================================= #
class Integrator:
    """
    Integrador de ecuaciones diferenciales
    """
    def __init__(self, model, odespy_method=odespy.RK4):
        self.model = model
        self.odespy_method = odespy_method

    def integrate_odespy(self, initial_conditions, time_points):  # -> pd.DataFrame()
        solver = self.odespy_method(self.model.odespy_func())
        solver.set_initial_condition(initial_conditions)
        u, t = solver.solve(time_points)
        sol = pd.DataFrame(u, columns=self.model.var_names, index=t)
        sol.index.name = 't'
        return sol

    @staticmethod
    def integrate_solver(model=LorenzAttractor, Xinit=np.array([1, 1, 1]), time_steps=1000, integration_dt=0.01):
        def ode_func(x, t):
            return model.get_dt(x)
        X_trayectory = odeint(ode_func, Xinit, t=np.linspace(0, time_steps*integration_dt, time_steps))
        return X_trayectory

    @staticmethod
    def integrate(model=LorenzAttractor, Xinit=np.array([1, 1, 1]), time_steps=1000, integration_dt=0.01):
        X_trayectory = np.zeros((time_steps, len(Xinit)))
        X_trayectory[0, :] = Xinit
        for i in range(1, time_steps):
            Xt = model.get_dt(X_trayectory[i - 1, :])
            Xtt = model.get_dtt(X_trayectory[i - 1, :])

            X_trayectory[i, :] = Xtt * integration_dt ** 2 / 2 + Xt * integration_dt + X_trayectory[i - 1, :]

        return X_trayectory