#File for exponential decay equation:
# u_t(x, t, t) = u_xx(x, t, t) - u_yy(x, t, t) - u(x, t, t)^3 => u(x, y, t) = C * sinx * siny * e^-t;
'''
Область определения:
x ∈ (0, pi)
y ∈ (0, pi)
t ∈ (0, 1)
Граничные условия:
На x = 0: u = 0
На x = pi: u = 0
На y = 0: u = exp(-t)*sin(x)
На y = pi: u = -exp(-t)*sin(x)
На t = 0: u = sin(x)*cos(y)
'''
import numpy as np
import matplotlib.pyplot as plt
import torch

#Linear, 1st order, ODE
class Nonlinear():
    # C means const in x(0); N - number of steps; a,b - borders (a<b)
    def __init__(self, C = 1, 
                 t_start = 0, t_finish = 0.2, t_dots = 100, 
                 grid_size = 21, 
                 x_start = 0, x_finish = np.pi,
                 y_start = 0, y_finish = np.pi,
                 ):
        self.C = C
        # time linspace
        self.t_start = t_start
        self.t_finish = t_finish
        self.t_dots = t_dots
        self.t = np.linspace(self.t_start, self.t_finish, self.t_dots)
        
        #grid linspace
        self.x_start = x_start
        self.x_finish = x_finish
        self.x_dots = grid_size
        self.x = np.linspace(self.x_start, self.x_finish, self.x_dots)
        
        
        self.y_start = y_start
        self.y_finish = y_finish
        self.y_dots = grid_size
        self.y = np.linspace(self.y_start, self.y_finish, self.y_dots)
        
        self.x, self.y, self.t = np.meshgrid(self.x, self.y, self.t)
        
        self.u = self.C * np.sin(self.x) * np.cos(self.y) * np.exp(-self.t)
        
    def equation(self, x: np.array, y: np.array, t: np.array) -> np.array:
        return self.C * np.sin(x) * np.cos(y) * np.exp(-t)
    
    def equation_torch(self, x: torch.tensor, y: torch.tensor, t: torch.tensor) -> torch.tensor:
        return self.C * torch.sin(x) * torch.cos(y) * torch.exp(-t)
        
    def get_points_at_time(self, t = 0):
        x = np.linspace(self.x_start, self.x_finish, self.x_dots)
        y = np.linspace(self.y_start, self.y_finish, self.y_dots)
        x, y, t = np.meshgrid(x, y, t)
        u = self.C * np.sin(x) * np.cos(y) * np.exp(-t)
        return u, x, y, t

    def plot(self):
        plt.plot(self.t, self.x, label = f'x(t) = {self.C} * e^-t')
        plt.xlabel('Time')
        plt.ylabel('x(t)')
        plt.legend()
        plt.show()
