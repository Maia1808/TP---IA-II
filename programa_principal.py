
import numpy as np
from controlador_pendulo import FuzzyController, FuzzySet, FuzzyVariable, FuzzyRule
import matplotlib.pyplot as plt

# Constantes del sistema
M = 1.0     # masa del carro (kg)
m = 0.3     # masa del péndulo (kg)
l = 0.3     # longitud del péndulo (m)
g = 9.81    # gravedad (m/s^2)
dt = 0.05   # paso de tiempo (s)

# Condiciones iniciales
theta = -45  # grados
theta_dot = 4  # rad/s

# Crear variables y controlador como antes
theta_sets = [
    FuzzySet("NG", -90, -90, -60),
    FuzzySet("NP", -70, -45, -20),
    FuzzySet("Z", -30, 0, 30),
    FuzzySet("PP", 20, 45, 70),
    FuzzySet("PG", 60, 90, 90),
]

theta_dot_sets = [
    FuzzySet("NG", -10, -10, -6),
    FuzzySet("NP", -8, -5, -2),
    FuzzySet("Z", -3, 0, 3),
    FuzzySet("PP", 2, 5, 8),
    FuzzySet("PG", 6, 10, 10),
]

force_sets = [
    FuzzySet("NG", -30, -30, -20),
    FuzzySet("NP", -25, -15, -5),
    FuzzySet("Z", -10, 0, 10),
    FuzzySet("PP", 5, 15, 25),
    FuzzySet("PG", 20, 30, 30),
]

rules_data = [
    ("NG", "NG", "NG"), ("NP", "NG", "NP"), ("Z", "NG", "NP"), ("PP", "NG", "NP"), ("PG", "NG", "Z"),
    ("NG", "NP", "NG"), ("NP", "NP", "NP"), ("Z", "NP", "NP"), ("PP", "NP", "Z"), ("PG", "NP", "PP"),
    ("NG", "Z", "NG"), ("NP", "Z", "NP"), ("Z", "Z", "Z"), ("PP", "Z", "PP"), ("PG", "Z", "PG"),
    ("NG", "PP", "NP"), ("NP", "PP", "Z"), ("Z", "PP", "PP"), ("PP", "PP", "PP"), ("PG", "PP", "PG"),
    ("NG", "PG", "Z"), ("NP", "PG", "PP"), ("Z", "PG", "PP"), ("PP", "PG", "PP"), ("PG", "PG", "PG"),
]

rules = [FuzzyRule(a1, a2, c) for a1, a2, c in rules_data]

theta_var = FuzzyVariable("theta", theta_sets)
theta_dot_var = FuzzyVariable("theta_dot", theta_dot_sets)
force_var = FuzzyVariable("force", force_sets)
controller = FuzzyController(theta_var, theta_dot_var, force_var, rules)

# Simulación
N = int(5 / dt)
theta_hist = []
theta_dot_hist = []
force_hist = []
time = []

for step in range(N):
    t = step * dt
    theta = max(-89.9, min(89.9, theta))  # limitar rango
    theta_dot = max(-9.9, min(9.9, theta_dot))

    F = controller.infer(theta, theta_dot)

    # Calcular aceleración angular (theta'')
    theta_rad = np.radians(theta)
    num = g * np.sin(theta_rad) + np.cos(theta_rad) * (
        (-F - m * l * theta_dot**2 * np.sin(theta_rad)) / (M + m)
    )
    denom = (4/3) - (m * np.cos(theta_rad)**2 / (M + m))
    theta_ddot = num / (l * denom)

    # Actualizar estado
    theta_new = theta + theta_dot * dt + 0.5 * theta_ddot * dt**2
    theta_dot_new = theta_dot + theta_ddot * dt

    # Guardar valores
    theta_hist.append(theta)
    theta_dot_hist.append(theta_dot)
    force_hist.append(F)
    time.append(t)

    theta = theta_new
    theta_dot = theta_dot_new

# Graficar resultados
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time, theta_hist)
plt.ylabel("θ (grados)")
plt.grid()
plt.title("Evolución del sistema carro-péndulo")

plt.subplot(3, 1, 2)
plt.plot(time, theta_dot_hist)
plt.ylabel("θ' (rad/s)")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, force_hist)
plt.ylabel("F (N)")
plt.xlabel("Tiempo (s)")
plt.grid()

plt.tight_layout()
plt.show()
