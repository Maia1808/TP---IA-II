import matplotlib.pyplot as plt
import numpy as np

# Función para visualización en tiempo real de theta y F
def visualizar_en_tiempo_real(theta_vals, fuerza_vals, tiempo_vals):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    line1, = ax1.plot([], [], label='θ (°)')
    ax1.set_xlim(tiempo_vals[0], tiempo_vals[-1])
    ax1.set_ylim(min(theta_vals)-10, max(theta_vals)+10)
    ax1.set_ylabel("Ángulo θ (°)")
    ax1.grid(True)

    line2, = ax2.plot([], [], label='F (N)', color='orange')
    ax2.set_xlim(tiempo_vals[0], tiempo_vals[-1])
    ax2.set_ylim(min(fuerza_vals)-5, max(fuerza_vals)+5)
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Fuerza aplicada F (N)")
    ax2.grid(True)

    for i in range(len(tiempo_vals)):
        line1.set_data(tiempo_vals[:i+1], theta_vals[:i+1])
        line2.set_data(tiempo_vals[:i+1], fuerza_vals[:i+1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    plt.ioff()
    plt.show()
