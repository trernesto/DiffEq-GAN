import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_solution(solution, domain, t_value=0):
    """
    Построение графиков аналитического решения.
    
    :param solution: аналитическое решение (символьное выражение)
    :param domain: область определения
    :param t_value: значение времени для визуализации
    """
    # Подготовка сетки
    x_vals = np.linspace(float(domain["x"][0]), float(domain["x"][1]), 100)
    y_vals = np.linspace(float(domain["y"][0]), float(domain["y"][1]), 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Вычисление решения на сетке
    t = sp.Symbol('t')
    x, y = sp.symbols('x y')
    u_func = sp.lambdify((x, y, t), solution, modules='numpy')
    U = u_func(X, Y, t_value)
    
    # Построение 3D-графика
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, U, cmap='viridis', edgecolor='k')
    ax.set_title(f"Аналитическое решение при t = {t_value}", fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y, t)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()
    
def plot_solution_heatmap(solution, domain, t_value=0):
    """
    Построение 2D-графика аналитического решения в виде heatmap.
    
    :param solution: аналитическое решение (символьное выражение)
    :param domain: область определения
    :param t_value: значение времени для построения heatmap
    """
    # Подготовка сетки
    x_vals = np.linspace(float(domain["x"][0]), float(domain["x"][1]), 100)
    y_vals = np.linspace(float(domain["y"][0]), float(domain["y"][1]), 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Преобразование аналитического решения в числовую функцию
    t = sp.Symbol('t')
    x, y = sp.symbols('x y')
    u_func = sp.lambdify((x, y, t), solution, modules='numpy')
    
    # Вычисление значений решения на сетке
    U = u_func(X, Y, t_value)
    
    # Построение heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, U, levels=100, cmap='viridis')
    plt.colorbar(label="u(x, y, t)")
    plt.title(f"Heatmap аналитического решения при t = {t_value}", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.grid(False)
    plt.show()
