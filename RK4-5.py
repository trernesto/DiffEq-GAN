import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from data.nonlinear import generate_pde_with_boundary_conditions
import sympy as sp

def solve_pde_rk45(solution, domain, time_span, grid_size=50):
    # Подготовка сетки
    x_vals = np.linspace(float(domain["x"][0]), float(domain["x"][1]), grid_size)
    y_vals = np.linspace(float(domain["y"][0]), float(domain["y"][1]), grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    initial_conditions = np.sin(X) * np.cos(Y)  # Начальное условие
    
    # Определяем правую часть уравнения
    def pde_rhs(t, u):
        u = u.reshape((grid_size, grid_size))
        # Вычисляем производные по x и y
        u_xx = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / (x_vals[1] - x_vals[0])**2
        u_yy = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / (y_vals[1] - y_vals[0])**2
        # Нелинейный член
        u_t = (u_xx - u_yy - u**3) #/ (u ** 2 + 1e-9)
        return u_t.flatten()

    # Решение методом Рунге-Кутта 4-5 порядка
    solution = solve_ivp(
        pde_rhs,
        time_span,
        initial_conditions.flatten(),
        method="RK45",
        t_eval=np.linspace(time_span[0], time_span[1], 100),
    )
    
    return X, Y, solution

# Параметры задачи
time_span = (0, 1)  # t ∈ [0, 1]
grid_size = 100

# Решение уравнения
analytical_solution, _, domain, _ = generate_pde_with_boundary_conditions()
X, Y, numerical_solution = solve_pde_rk45(analytical_solution, domain, time_span, grid_size)

# Визуализация результата
def plot_heatmap_at_t(X, Y, solution, t_index):
    """
    Построение heatmap для численного решения на фиксированном временном слое.
    
    :param X: координаты по x
    :param Y: координаты по y
    :param solution: массив решения
    :param t_index: индекс временного слоя
    """
    U_t = solution.y[:, t_index].reshape(X.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, U_t, levels=100, cmap="viridis")
    plt.colorbar(label="u(x, y, t)")
    plt.title(f"Численное решение при t = {solution.t[t_index]:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# Построение графиков
#plot_heatmap_at_t(X, Y, numerical_solution, t_index=0)  # Начальное состояние
#plot_heatmap_at_t(X, Y, numerical_solution, t_index=1)  # Конечное состояние


def calculate_errors(numerical_solution, analytical_solution, X, Y, domain):
    """
    Вычисление максимальной абсолютной и средней квадратичной погрешности.
    
    :param numerical_solution: численное решение (результат solve_ivp)
    :param analytical_solution: аналитическое решение (символьное выражение)
    :param X: сетка по x
    :param Y: сетка по y
    :param domain: область определения
    :return: список ошибок на каждом временном слое
    """
    t_vals = numerical_solution.t
    num_solutions = numerical_solution.y.T.reshape(len(t_vals), *X.shape)
    
    # Преобразуем аналитическое решение в числовую функцию
    x, y, t = sp.symbols('x y t')
    u_exact_func = sp.lambdify((x, y, t), analytical_solution, modules='numpy')
    
    # Вычисляем аналитическое решение на каждом временном слое
    errors = []
    for i, t_val in enumerate(t_vals):
        U_exact = u_exact_func(X, Y, t_val)
        U_num = num_solutions[i]
        
        # Погрешности
        max_error = np.max(np.abs(U_num - U_exact))
        mse = np.sqrt(np.mean((U_num - U_exact)**2))
        errors.append((t_val, max_error, mse))
    
    return errors

# Рассчитаем погрешности
errors = calculate_errors(numerical_solution, analytical_solution, X, Y, domain)

# Выводим результаты
print("Временной слой  |  Максимальная ошибка  |  Среднеквадратичная ошибка")
for t_val, max_error, mse in errors:
    print(f"{t_val:>14.3f}  |  {max_error:>22.6f}  |  {mse:>24.6f}")

# Визуализация ошибок
def plot_errors(errors):
    """
    Визуализация ошибок во времени.
    
    :param errors: список ошибок [(t, max_error, mse), ...]
    """
    times = [e[0] for e in errors]
    max_errors = [e[1] for e in errors]
    mse_errors = [e[2] for e in errors]
    
    plt.figure(figsize=(10, 5))
    plt.plot(times, max_errors, label="Максимальная ошибка", marker='o')
    plt.plot(times, mse_errors, label="Среднеквадратичная ошибка", marker='s')
    plt.xlabel("Время (t)", fontsize=12)
    plt.ylabel("Ошибка", fontsize=12)
    plt.title("Погрешности численного решения", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

# Построение графиков ошибок
plot_errors(errors)
