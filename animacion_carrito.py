import pygame
import numpy as np

# Dimensiones de la ventana
ANCHO_VENTANA = 1920
ALTO_VENTANA = 1080
ESCALA = 200  # Escala en píxeles por metro
FACTOR_DESPLAZAMIENTO = 60 # Factor para hacer más notorio el desplazamiento

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
AZUL = (0, 0, 255)
ROJO = (255, 0, 0)
VERDE = (0, 255, 0)

# Fuente para la información en pantalla
pygame.font.init()
FUENTE = pygame.font.Font(None, 36)

# Función para visualizar la simulación en tiempo real
def visualizar_en_tiempo_real(y_theta, y_fuerza, tiempos, L=0.5, dt=0.01, m_carrito=1):
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
    pygame.display.set_caption("Simulación del péndulo invertido")
    reloj = pygame.time.Clock()

    # Posiciones iniciales
    carrito_x = ANCHO_VENTANA // 2
    carrito_y = ALTO_VENTANA - 150  # Un poco arriba del piso
    carrito_ancho = 80
    carrito_alto = 40

    # Variables para el movimiento
    velocidad_carrito = 0  # Velocidad inicial del carrito
    aceleracion_carrito = 0  # Aceleración inicial

    for i, theta_deg in enumerate(y_theta):
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                return

        pantalla.fill(BLANCO)

        # Dibujar el piso debajo del carrito
        pygame.draw.line(pantalla, NEGRO, 
                 (0, carrito_y + carrito_alto // 2 ), 
                 (ANCHO_VENTANA, carrito_y + carrito_alto // 2), 5)

        # Dibujar la línea del piso debajo del carrito (justo en la base del carrito)
        pygame.draw.line(pantalla, NEGRO, 
                         (carrito_x - carrito_ancho // 2, carrito_y + carrito_alto // 2), 
                         (carrito_x + carrito_ancho // 2, carrito_y + carrito_alto // 2), 5)

        # Carrito
        pygame.draw.rect(pantalla, AZUL, (carrito_x - carrito_ancho // 2, carrito_y - carrito_alto // 2, carrito_ancho, carrito_alto))

        # Pértiga
        theta_rad = np.radians(theta_deg)
        x_final = carrito_x + L * ESCALA * np.sin(theta_rad)
        y_final = carrito_y - L * ESCALA * np.cos(theta_rad)

        pygame.draw.line(pantalla, AZUL, (carrito_x, carrito_y), (x_final, y_final), 5)
        pygame.draw.circle(pantalla, AZUL, (int(x_final), int(y_final)), 10)

        # Mostrar información sobre el ángulo, fuerza y tiempo
        texto_angulo = FUENTE.render(f"Ángulo: {np.degrees(theta_rad):.1f}°", True, NEGRO)
        texto_fuerza = FUENTE.render(f"Fuerza: {y_fuerza[i]:.2f} N", True, NEGRO)
        texto_tiempo = FUENTE.render(f"Tiempo: {tiempos[i]:.2f} s", True, NEGRO)

        pantalla.blit(texto_angulo, (20, 20))
        pantalla.blit(texto_fuerza, (20, 60))
        pantalla.blit(texto_tiempo, (20, 100))

        # Indicador de fuerza
        if y_fuerza[i] != 0:
            color_fuerza = ROJO
            direccion = 1 if y_fuerza[i] > 0 else -1
            longitud = min(100, abs(y_fuerza[i]) * 5)

            pygame.draw.line(pantalla, color_fuerza,
                            (carrito_x, carrito_y + 20),
                            (carrito_x + direccion * longitud, carrito_y + 20), 5)
            pygame.draw.polygon(pantalla, color_fuerza, [
                (carrito_x + direccion * longitud, carrito_y + 15),
                (carrito_x + direccion * longitud, carrito_y + 25),
                (carrito_x + direccion * (longitud + 10), carrito_y + 20)
            ])

        # Movimiento del carrito: usar la aceleración derivada de la fuerza
        # Cálculo de aceleración del carrito en función de la fuerza
        aceleracion_carrito = y_fuerza[i] / m_carrito  # Suponiendo que la fuerza es la responsable de mover el carrito

        # Actualización de la velocidad del carrito con la aceleración
        velocidad_carrito += aceleracion_carrito * dt  # Velocidad por la aceleración

        # Actualización de la posición del carrito usando la fórmula de desplazamiento
        carrito_x += (velocidad_carrito * dt + 0.5 * aceleracion_carrito * dt**2) * FACTOR_DESPLAZAMIENTO  # Desplazamiento del carrito

        # Evitar que el carrito se salga de la pantalla
        if carrito_x - carrito_ancho // 2 < 0:
            carrito_x = carrito_ancho // 2
        elif carrito_x + carrito_ancho // 2 > ANCHO_VENTANA:
            carrito_x = ANCHO_VENTANA - carrito_ancho // 2

        pygame.display.flip()
        reloj.tick(60)  # 60 FPS

    pygame.quit()
