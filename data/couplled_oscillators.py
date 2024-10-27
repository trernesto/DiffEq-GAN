#File for exponential coupled oscillators equation
# System {x'(t) = -ty; y'(t) = tx} => x(t) = C1 * C2 * cos(t^2/2), y(t) = C1 * C2 * sin(t^2/2)
# x(0) = a, y(0) = 0 => C1 * C2 = a
# x(sqrt(pi)) = 0, y(sqrt(pi)) = b => C1 * C2 = b 

import numpy as np
import matplotlib.pyplot as plt

#Linear, 1st order, ODE
class CoupOsc():
    # x0 means x(0); N - number of steps; a,b - borders (a<b)
    def __init__(self, C = 1, a = 0, b = 2 * np.pi, N = 1001):
        self.t = np.linspace(a, b, N)
        self.C = C
        self.x = self.C * np.cos(self.t ** 2 / 2)
        self.y = self.C * np.sin(self.t ** 2 / 2)

    def plot(self):
        plt.plot(self.t, self.x, label = f'x(t) = {self.C} * cos(t^2 / 2)')
        plt.plot(self.t, self.y, label = f'y(t) = {self.C} * sin(t^2 / 2)')
        plt.xlabel('Time')
        plt.ylabel('func(t)')
        plt.legend()
        plt.show()
