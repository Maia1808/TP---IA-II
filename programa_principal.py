
import numpy as np
from controlador_pendulo import crear_controlador
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

controller= crear_controlador()

# Simulación por 5 segundos
N = int(5 / dt)
theta_hist = []
theta_dot_hist = []
force_hist = []
time = []

for step in range(N):
    t = step * dt
    theta = max(-89.9, min(89.9, theta))
    theta_dot = max(-9.9, min(9.9, theta_dot))

    F = controller.infer(theta, theta_dot)

    theta_rad = np.radians(theta)
    num = g * np.sin(theta_rad) + np.cos(theta_rad) * (
        (-F - m * l * theta_dot**2 * np.sin(theta_rad)) / (M + m)
    )
    denom = (4/3) - (m * np.cos(theta_rad)**2 / (M + m))
    theta_ddot = num / (l * denom)

    theta_new = theta + theta_dot * dt + 0.5 * theta_ddot * dt**2
    theta_dot_new = theta_dot + theta_ddot * dt

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
