import sympy as sp
#from plot import plot_solution

def generate_pde_with_boundary_conditions():
    """
    Генерация нелинейного ДНУЧП с аналитическим решением и граничными условиями.
    """
    # Определяем независимые переменные
    x, y, t = sp.symbols('x y t')
    
    # Задаём аналитическое решение
    solution = sp.sin(x) * sp.cos(y) * sp.exp(-t)
    
    u = solution
    # Вычисляем частные производные
    u_x = sp.diff(solution, x)
    u_y = sp.diff(solution, y)
    u_t = sp.diff(solution, t)
    u_xx = sp.diff(u_x, x)
    u_yy = sp.diff(u_y, y)
    
    # Формируем ДНУЧП
    pde = sp.Eq(u**2 * u_t, u_xx - u_yy - u**3)
    
    # Определяем область и граничные условия
    domain = {
        "x": (0, sp.pi),   # x ∈ [0, π]
        "y": (0, sp.pi),   # y ∈ [0, π]
        "t": (0, 1)        # t ∈ [0, 1]
    }
    
    boundary_conditions = {
        # Граничные условия на x = 0
        (x, 0): solution.subs(x, 0),
        # Граничные условия на x = π
        (x, sp.pi): solution.subs(x, sp.pi),
        # Граничные условия на y = 0
        (y, 0): solution.subs(y, 0),
        # Граничные условия на y = π
        (y, sp.pi): solution.subs(y, sp.pi),
        # Начальное условие для t = 0
        (t, 0): solution.subs(t, 0)
    }
    
    return solution, pde, domain, boundary_conditions

# Пример использования
analytical_solution, pde, domain, boundary_conditions = generate_pde_with_boundary_conditions()



# Пример использования
analytical_solution, pde, domain, boundary_conditions = generate_pde_with_boundary_conditions()

print("Аналитическое решение:")
sp.pprint(analytical_solution)
print("Сгенерированное ДНУЧП:")
sp.pprint(pde)
print("Область определения:")
for var, bounds in domain.items():
    print(f"{var} ∈ {bounds}")
print("Граничные условия:")
for (var, val), condition in boundary_conditions.items():
    print(f"На {var} = {val}: u = {condition}")
    
# Построение графика для t = 0 и t = 0.5
#plot_solution(analytical_solution, domain, t_value=0)
#plot_solution(analytical_solution, domain, t_value=0.5)