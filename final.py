import numpy as np
import pygame

# Parámetros del sistema físico
g = 9.81    # gravedad (m/s^2)
l = 0.5     # longitud del péndulo (m)
m = 0.1     # masa del péndulo (kg)
M = 1.0     # masa del carrito (kg)
dt = 0.05   # paso de simulación (s)

# Parámetros del controlador difuso
etiquetas = ['NG', 'NP', 'Z', 'PP', 'PG']
dominio_e = np.linspace(-0.5, 0.5, 100)
dominio_de = np.linspace(-1.5, 1.5, 100)
dominio_f = np.linspace(-10, 10, 100)

# Función de pertenencia triangular
def triangular(x, a, b, c):
    return np.maximum(np.minimum((x - a)/(b - a), (c - x)/(c - b)), 0)

# Función de pertenencia trapezoidal
def trapezoidal(x, a, b, c, d):
    return np.maximum(np.minimum(np.minimum((x - a)/(b - a), 1), (d - x)/(d - c)), 0)

# Generar funciones de pertenencia
def generar_fm(dominio, tipo='triangular'):
    n = len(etiquetas)
    ancho = dominio[-1] - dominio[0]
    paso = ancho / (n - 1)
    centros = [dominio[0] + i * paso for i in range(n)]
    fms = []

    for i, c in enumerate(centros):
        if i == 0:
            a, b, c_, d = c - paso, c - paso, c, c + paso
            fm = trapezoidal(dominio, a, b, c_, d)
        elif i == n - 1:
            a, b, c_, d = c - paso, c, c + paso, c + paso
            fm = trapezoidal(dominio, a, b, c_, d)
        else:
            a, b, c_ = c - paso, c, c + paso
            fm = triangular(dominio, a, b, c_)

        fms.append(fm)

    return np.array(fms)

# Reglas difusas
reglas = [
    ('NG', 'NG', 'PG'), ('NG', 'NP', 'PG'), ('NG', 'Z', 'PG'),  ('NG', 'PP', 'PP'), ('NG', 'PG', 'Z'),
    ('NP', 'NG', 'PG'), ('NP', 'NP', 'PG'), ('NP', 'Z', 'PP'),  ('NP', 'PP', 'Z'),  ('NP', 'PG', 'NP'),
    ('Z',  'NG', 'PP'), ('Z',  'NP', 'PP'), ('Z',  'Z', 'Z'),   ('Z',  'PP', 'NP'), ('Z',  'PG', 'NP'),
    ('PP', 'NG', 'NP'), ('PP', 'NP', 'NP'), ('PP', 'Z', 'NP'),  ('PP', 'PP', 'PG'), ('PP', 'PG', 'PG'),
    ('PG', 'NG', 'Z'),  ('PG', 'NP', 'NP'), ('PG', 'Z', 'PP'),  ('PG', 'PP', 'PG'), ('PG', 'PG', 'PG'),
]

# Función de inferencia difusa
def inferencia_difusa(error, d_error, fms_e, fms_de, fms_f):
    grados_e = [triangular(error, *np.interp([0, 0.5, 1], [0, 1, 2], [dominio_e[0], dominio_e[len(dominio_e)//2], dominio_e[-1]])) for fm in fms_e]
    grados_de = [triangular(d_error, *np.interp([0, 0.5, 1], [0, 1, 2], [dominio_de[0], dominio_de[len(dominio_de)//2], dominio_de[-1]])) for fm in fms_de]

    salida_agregada = np.zeros_like(dominio_f)
    for (e_lbl, de_lbl, f_lbl) in reglas:
        i_e = etiquetas.index(e_lbl)
        i_de = etiquetas.index(de_lbl)
        i_f = etiquetas.index(f_lbl)
        peso = min(grados_e[i_e], grados_de[i_de])
        salida_agregada = np.fmax(salida_agregada, np.fmin(peso, fms_f[i_f]))

    if np.sum(salida_agregada) == 0:
        return 0.0
    return np.sum(salida_agregada * dominio_f) / np.sum(salida_agregada)

# Generar funciones de pertenencia
fms_e = generar_fm(dominio_e)
fms_de = generar_fm(dominio_de)
fms_f = generar_fm(dominio_f)

# Inicialización del sistema
x, x_dot = 0.0, 0.0
theta, theta_dot = 0.2, 0.0
trayectoria = []

# Simulación física con control difuso
for _ in range(400):
    error = theta
    d_error = theta_dot
    fuerza = inferencia_difusa(error, d_error, fms_e, fms_de, fms_f)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    total_mass = m + M
    temp = (fuerza + m * l * theta_dot**2 * sin_theta) / total_mass
    theta_acc = (g * sin_theta - cos_theta * temp) / (l * (4/3 - m * cos_theta**2 / total_mass))
    x_acc = temp - m * l * theta_acc * cos_theta / total_mass

    x += x_dot * dt
    x_dot += x_acc * dt
    theta += theta_dot * dt
    theta_dot += theta_acc * dt

    trayectoria.append((x, theta))

# --------------------------
# Visualización con Pygame
# --------------------------
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Péndulo Invertido")
clock = pygame.time.Clock()

pixels_per_meter = 100
car_width = 60
car_height = 30
pendulum_length = l * pixels_per_meter

def sim_to_screen(x_sim, y_sim):
    return int(WIDTH / 2 + x_sim * pixels_per_meter), int(HEIGHT / 2 - y_sim)

running = True
index = 0

while running:
    screen.fill((255, 255, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if index < len(trayectoria):
        x, theta = trayectoria[index]
        index += 1
    else:
        index = 0

    car_x, car_y = sim_to_screen(x, 0)
    pygame.draw.rect(screen, (0, 100, 200), (car_x - car_width // 2, car_y - car_height // 2, car_width, car_height))

    pendulum_x = car_x + pendulum_length * np.sin(theta)
    pendulum_y = car_y + pendulum_length * np.cos(theta)
    pygame.draw.line(screen, (200, 50, 50), (car_x, car_y), (pendulum_x, pendulum_y), 5)
    pygame.draw.circle(screen, (0, 0, 0), (int(pendulum_x), int(pendulum_y)), 8)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
