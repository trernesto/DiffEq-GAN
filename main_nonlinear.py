import sympy as sp
from data.nonlinear import generate_pde_with_boundary_conditions
from plot import plot_solution
from plot import plot_solution_heatmap

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
plot_solution(analytical_solution, domain, t_value=0)
#plot_solution(analytical_solution, domain, t_value=0.5)

plot_solution_heatmap(analytical_solution, domain, t_value=0)