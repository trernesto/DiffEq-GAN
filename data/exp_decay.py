#File for exponential decay equation:
# x'(t) + x(t) = 0 => x(t) = C * e^-t;
# if x(0) = 1 => x(t) = e^-t
import numpy as np
import matplotlib.pyplot as plt

#Linear, 1st order, ODE
class ExpDecay():
    # C means const in x(0); N - number of steps; a,b - borders (a<b)
    def __init__(self, C = 1, a = 0, b = 10, N = 1001):
        self.a = a
        self.b = b
        self.N = N
        self.t = np.linspace(a, b, N)
        self.C = C
        self.x = self.C * np.exp(-self.t)
        
    def equation(self, t: np.array) -> np.array:
        return self.C * np.exp(-t)
        
    def set_points_to_random(self, C_start = 1, C_finish = 1):
        a = self.a
        b = self.b
        N = self.N
        self.t = np.random.uniform(a, b, size = N)
        #self.C = C
        self.C = np.random.uniform(C_start, C_finish, size=N)
        self.x = self.equation(self.t)

    def plot(self):
        plt.plot(self.t, self.x, label = f'x(t) = {self.C} * e^-t')
        plt.xlabel('Time')
        plt.ylabel('x(t)')
        plt.legend()
        plt.show()
