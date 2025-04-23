
import numpy as np
import matplotlib.pyplot as plt

class FuzzySet:
    def __init__(self, name, *points):
        self.name = name # Nombre del conjunto difuso
        self.points = points

    def membership(self, x):
        if len(self.points) == 3: # Triangular 
            a, b, c = self.points
            if x <= a or x >= c: 
                return 0
            elif a < x <= b:
                return (x - a) / (b - a) #Es semejanza de triangulos
            elif b < x < c:
                return (c - x) / (c - b)
            elif x == b:
                return 1.0
            
        elif len(self.points) == 4: # para los extremos
            a, b, c, d = self.points
            
            if b < 0:  # medio trapecio izquierdo (ej: NG)
                if  x >= d:
                    return 0
                if a <= x <= b:
                    return 1  # plano horizontal
                elif c < x < d:
                    return (d - x) / (d - c)  # pendiente negativa
                
            elif b > 0:  # medio trapecio derecho (ej: PG)
                if  x <= a:
                    return 0
                if a < x < b:
                    return (x - a) / (b - a)  # pendiente positiva
                elif c <= x <= d:
                    return 1  # plano horizontal
                else:
                    return 0
        return 0 # porque si no se chotea

class FuzzyVariable:
    def __init__(self, name, sets):
        self.name = name
        self.sets = {s.name: s for s in sets} #convierte lista a diccionario

    def fuzzify(self, x):
        return {name: fs.membership(x) for name, fs in self.sets.items()}

class FuzzyRule:
    def __init__(self, antecedent1, antecedent2, consequent):
        self.antecedent1 = antecedent1 #conjunto de tita (ej: NP)
        self.antecedent2 = antecedent2 #conjunto de tita' (ej:PG)
        self.consequent = consequent #conjunto de F (ej: Z)

    def evaluate(self, theta_memberships, theta_dot_memberships):
        μ1 = theta_memberships[self.antecedent1]
        μ2 = theta_dot_memberships[self.antecedent2]
        return min(μ1, μ2), self.consequent

class FuzzyController:
    def __init__(self, theta_var, theta_dot_var, force_var, rules):
        self.theta_var = theta_var # posicion
        self.theta_dot_var = theta_dot_var #velocidad
        self.force_var = force_var # fuerza
        self.rules = rules
        self.last_output_memberships = {} #para desp graficar
        self.last_centroid = 0
        self.last_inputs = (0, 0)

    def infer(self, theta_val, theta_dot_val):

        # Aplicar circularidad a theta
        if theta_val > np.pi:
            theta_val -= 2 * np.pi
        elif theta_val < -np.pi:
            theta_val += 2 * np.pi

        self.last_inputs = (theta_val, theta_dot_val) # para graficar
        μ_theta = self.theta_var.fuzzify(theta_val)
        μ_theta_dot = self.theta_dot_var.fuzzify(theta_dot_val)

        output_memberships = {x: 0 for x in np.linspace(-30, 30, 1000)}

        for rule in self.rules: # crea el eje de salida de F con 1000 puntos
            activation, label = rule.evaluate(μ_theta, μ_theta_dot)
            fuzzy_set = self.force_var.sets[label]
            for x in output_memberships:
                μ = min(activation, fuzzy_set.membership(x))
                output_memberships[x] = max(output_memberships[x], μ)
        
        self.last_output_memberships = output_memberships #para desp graficar
        
        # Defuzzificación por el método del centroide
        num = sum(x * μ for x, μ in output_memberships.items())
        den = sum(μ for μ in output_memberships.values())
        self.last_centroid = num / den if den != 0 else 0
        return self.last_centroid
    
    
    def graficar_resultado(self):
        theta_val, theta_dot_val = self.last_inputs
        x_force = np.linspace(-30, 30, 1000)
        x_theta = np.linspace(-np.pi, np.pi, 1000)
        x_theta_dot = np.linspace(-10, 10, 1000)

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # theta
        for label, fs in self.theta_var.sets.items():
            axs[0].plot(x_theta, [fs.membership(x) for x in x_theta], linestyle='--', label=label)
        axs[0].axvline(theta_val, color='red', linestyle='-', label=f'theta = {theta_val}')
        axs[0].set_title("Funciones de pertenencia de θ (posición)")
        axs[0].set_xlabel("θ (grados)")
        axs[0].set_ylabel("μ")
        axs[0].grid(True)
        axs[0].legend()

        # theta'
        for label, fs in self.theta_dot_var.sets.items():
            axs[1].plot(x_theta_dot, [fs.membership(x) for x in x_theta_dot], linestyle='--', label=label)
        axs[1].axvline(theta_dot_val, color='red', linestyle='-', label=f"θ' = {theta_dot_val}")
        axs[1].set_title("Funciones de pertenencia de θ' (velocidad angular)")
        axs[1].set_xlabel("θ' (rad/s)")
        axs[1].set_ylabel("μ")
        axs[1].grid(True)
        axs[1].legend()

        # fuerza combinada
        for label, fs in self.force_var.sets.items():
            axs[2].plot(x_force, [fs.membership(x) for x in x_force], linestyle='--', label=f"{label} original")
        y_combined = [self.last_output_memberships.get(x, 0) for x in x_force]
        axs[2].plot(x_force, y_combined, color='black', linewidth=2.5, label="Salida combinada")
        axs[2].axvline(self.last_centroid, color='red', linestyle='-', label=f"Centroide = {self.last_centroid:.2f}")
        axs[2].set_title("Funciones de pertenencia de la fuerza y salida combinada")
        axs[2].set_xlabel("Fuerza (N)")
        axs[2].set_ylabel("μ")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        plt.show()

# Ejemplo de definición de variables y conjuntos
def crear_controlador():
    theta_sets = [
        FuzzySet("NG", -np.pi, -2.5, -2.5, -1.5),
        FuzzySet("NP", -2, -1, 0),
        FuzzySet("Z", -0.5, 0, 0.5),
        FuzzySet("PP", 0, 1, 2),
        FuzzySet("PG", 1.5, 2.5, 2.5, np.pi),
    ]
    theta_dot_sets = [
        FuzzySet("NG", -10, -7.5, -7.5, -5),
        FuzzySet("NP", -7, -4, -1),
        FuzzySet("Z", -2, 0, 2),
        FuzzySet("PP", 1, 4, 7),
        FuzzySet("PG", 5, 7.5, 7.5, 10),
    ]
    force_sets = [
        FuzzySet("NG", -30, -28, -28, -20),
        FuzzySet("NP", -25, -15, -5),
        FuzzySet("Z", -10, 0, 10),
        FuzzySet("PP", 5, 15, 25),
        FuzzySet("PG", 20, 28, 28, 30),
    ]

    # Cargar reglas
    rules_data = [
            ("NG", "NG", "NG"), ("NP", "NG", "NG"), ("Z", "NG", "NP"), ("PP", "NG", "NP"), ("PG", "NG", "Z"),
            ("NG", "NP", "NG"), ("NP", "NP", "NP"), ("Z", "NP", "NP"), ("PP", "NP", "Z"), ("PG", "NP", "PP"),
            ("NG", "Z", "NG"), ("NP", "Z", "NP"), ("Z", "Z", "Z"), ("PP", "Z", "PP"), ("PG", "Z", "PG"),
            ("NG", "PP", "NP"), ("NP", "PP", "Z"), ("Z", "PP", "PP"), ("PP", "PP", "PP"), ("PG", "PP", "PG"),
            ("NG", "PG", "Z"), ("NP", "PG", "PP"), ("Z", "PG", "PG"), ("PP", "PG", "PG"), ("PG", "PG", "PG"),
    ]

    rules = [FuzzyRule(a1, a2, c) for a1, a2, c in rules_data]

    return FuzzyController(FuzzyVariable("theta", theta_sets), FuzzyVariable("theta_dot", theta_dot_sets), FuzzyVariable("force", force_sets), rules)
'''
#Ejemplo de uso
theta_input =  np.radians(150)  # grados
theta_dot_input = 0  # rad/s
controller= crear_controlador()
output_force = controller.infer(theta_input, theta_dot_input)
print("Fuerza resultante:", output_force)

#graficar
controller.graficar_resultado()
'''