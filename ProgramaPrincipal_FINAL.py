import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from controlador_pendulo_FINAL import crear_controlador
from visualizar import visualizar_en_tiempo_real


# Constantes físicas
CONSTANTE_M = 1 # Masa del carro
CONSTANTE_m = 0.1 # Masa de la pertiga
CONSTANTE_l = 0.5 # Longitud dela pertiga
dt=0.01

# Crear controlador difuso
controller = crear_controlador()

# Función para normalizar el ángulo al rango [-π, π]
def normalizar_angulo(angulo):
    return (angulo + np.pi) % (2 * np.pi) - np.pi

# Simula el modelo del carro-pendulo con lógica difusa
def simular(t_max, delta_t, theta_0, v_0, a_0):
    theta = np.radians(theta_0)  # radianes
    v = v_0
    a = a_0

    y_theta = []
    y_theta_dot = []
    y_fuerza = []
    x = np.arange(0, t_max, delta_t)

    for t in x:
        f = controller.infer(theta, v)
        a = calcula_aceleracion(theta, v, f)

        theta = theta + v * delta_t + a * (delta_t ** 2) / 2 ###
        v = v + a * delta_t
        

        # Normalizar el ángulo
        theta = normalizar_angulo(theta)

        y_theta.append(np.degrees(theta))
        y_theta_dot.append(v)
        y_fuerza.append(f)

    #para graficar en tiempo real
    #visualizar_en_tiempo_real(y_theta, y_fuerza, x)

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
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    num = constants.g * sin_theta + cos_theta * ((-f - CONSTANTE_m * CONSTANTE_l * v**2 * sin_theta) / (CONSTANTE_M + CONSTANTE_m))
    denom = CONSTANTE_l * (4/3 - (CONSTANTE_m * cos_theta**2) / (CONSTANTE_M + CONSTANTE_m))
    return num / denom if abs(denom) > 1e-6 else 0

# Ejecutar simulación
simular(t_max=5, delta_t=0.01, theta_0=180, v_0=0, a_0=0)
