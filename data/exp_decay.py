#File for exponential decay equation:
# x'(t) + x(t) = 0 => x(t) = C * e^-t;
# if x(0) = 1 => x(t) = e^-t
import numpy as np

#Linear, 1st order, ODE
class ExpDecay():
    # x0 means x(0); N - number of steps; a,b - borders (a<b)
    def __init__(self, x0 = 1, a = 0, b = 10, N = 1001):
        self.t = np.linspace(a, b, N)
        self.x = x0 * np.exp(-self.t)