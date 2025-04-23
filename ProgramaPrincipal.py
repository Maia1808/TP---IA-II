import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from controlador_pendulo import crear_controlador

# Constantes físicas
CONSTANTE_M = 2 # Masa del carro
CONSTANTE_m = 1 # Masa de la pertiga
CONSTANTE_l = 1 # Longitud dela pertiga

# Crear controlador difuso
controller = crear_controlador()

# Simula el modelo del carro-pendulo con lógica difusa
def simular(t_max, delta_t, theta_0, v_0, a_0):
    theta = (theta_0 * np.pi) / 180  # radianes
    v = v_0
    a = a_0

    y_theta = []
    y_theta_dot = []
    y_fuerza = []
    x = np.arange(0, t_max, delta_t)

    for t in x:
        theta_deg = np.degrees(theta)
        theta_dot = v
        f = controller.infer(theta_deg, theta_dot)
        a = calcula_aceleracion(theta, v, f)

        v = v + a * delta_t
        theta = theta + v * delta_t + a * (delta_t ** 2) / 2


        y_theta.append(np.degrees(theta))
        y_theta_dot.append(theta_dot)
        y_fuerza.append(f)

    # Graficar los 3 resultados en subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(x, y_theta)
    axs[0].set_title("Ángulo θ (grados)")
    axs[0].set_ylabel("θ (°)")
    axs[0].grid()

    axs[1].plot(x, y_theta_dot)
    axs[1].set_title("Velocidad angular θ'")
    axs[1].set_ylabel("θ' (rad/s)")
    axs[1].grid()

    axs[2].plot(x, y_fuerza)
    axs[2].set_title("Fuerza aplicada")
    axs[2].set_ylabel("F (N)")
    axs[2].set_xlabel("Tiempo (s)")
    axs[2].grid()

    plt.tight_layout()
    plt.show()


# Cálculo de aceleración angular θ'' con fuerza f
def calcula_aceleracion(theta, v, f):
    num = constants.g * np.sin(theta) + np.cos(theta) * (
        (-f - CONSTANTE_m * CONSTANTE_l * v**2 * np.sin(theta)) / (CONSTANTE_M + CONSTANTE_m)
    )
    denom = CONSTANTE_l * (4/3 - (CONSTANTE_m * np.cos(theta)**2 / (CONSTANTE_M + CONSTANTE_m)))
    return num / denom

# Ejecutar simulación
simular(t_max=10, delta_t=0.01, theta_0=-90, v_0=0, a_0=0)
