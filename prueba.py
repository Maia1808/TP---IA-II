import numpy as np
import matplotlib.pyplot as plt

class FuzzySet:
    def __init__(self, name, *points):
        self.name = name
        self.points = points

    def membership(self, x):
        if len(self.points) == 3:
            a, b, c = self.points
            if x <= a or x >= c:
                return 0
            elif a < x <= b:
                return (x - a) / (b - a)
            elif b < x < c:
                return (c - x) / (c - b)
            elif x == b:
                return 1.0
        elif len(self.points) == 4:
            a, b, c, d = self.points
            if x <= a or x >= d:
                return 0
            elif a < x < b:
                return (x - a) / (b - a)
            elif b <= x <= c:
                return 1
            elif c < x < d:
                return (d - x) / (d - c)

class FuzzyVariable:
    def __init__(self, name, sets):
        self.name = name
        self.sets = {s.name: s for s in sets}

    def fuzzify(self, x):
        return {name: fs.membership(x) for name, fs in self.sets.items()}

class FuzzyRule:
    def __init__(self, antecedent1, antecedent2, consequent):
        self.antecedent1 = antecedent1
        self.antecedent2 = antecedent2
        self.consequent = consequent

    def evaluate(self, theta_memberships, theta_dot_memberships):
        μ1 = theta_memberships[self.antecedent1]
        μ2 = theta_dot_memberships[self.antecedent2]
        return min(μ1, μ2), self.consequent

class FuzzyController:
    def __init__(self, theta_var, theta_dot_var, force_var, rules):
        self.theta_var = theta_var
        self.theta_dot_var = theta_dot_var
        self.force_var = force_var
        self.rules = rules
        self.last_output_memberships = {}
        self.last_centroid = 0
        self.last_inputs = (0, 0)

    def infer(self, theta_val, theta_dot_val):
        self.last_inputs = (theta_val, theta_dot_val)
        μ_theta = self.theta_var.fuzzify(theta_val)
        μ_theta_dot = self.theta_dot_var.fuzzify(theta_dot_val)

        output_memberships = {x: 0 for x in np.linspace(-31, 31, 1000)}

        for rule in self.rules:
            activation, label = rule.evaluate(μ_theta, μ_theta_dot)
            fuzzy_set = self.force_var.sets[label]
            for x in output_memberships:
                μ = min(activation, fuzzy_set.membership(x))
                output_memberships[x] = max(output_memberships[x], μ)

        self.last_output_memberships = output_memberships

        num = sum(x * μ for x, μ in output_memberships.items())
        den = sum(μ for μ in output_memberships.values())
        self.last_centroid = num / den if den != 0 else 0
        return self.last_centroid

    def graficar_resultado(self):
        theta_val, theta_dot_val = self.last_inputs
        x_force = np.linspace(-31, 31, 1000)
        x_theta = np.linspace(-91, 91, 1000)
        x_theta_dot = np.linspace(-11, 11, 1000)

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # θ
        for label, fs in self.theta_var.sets.items():
            axs[0].plot(x_theta, [fs.membership(x) for x in x_theta], linestyle='--', label=label)
        axs[0].axvline(theta_val, color='red', linestyle='-', label=f'theta = {theta_val}')
        axs[0].set_title("Funciones de pertenencia de θ (posición)")
        axs[0].set_xlabel("θ (grados)")
        axs[0].set_ylabel("μ")
        axs[0].grid(True)
        axs[0].legend()

        # θ'
        for label, fs in self.theta_dot_var.sets.items():
            axs[1].plot(x_theta_dot, [fs.membership(x) for x in x_theta_dot], linestyle='--', label=label)
        axs[1].axvline(theta_dot_val, color='red', linestyle='-', label=f"θ' = {theta_dot_val}")
        axs[1].set_title("Funciones de pertenencia de θ' (velocidad angular)")
        axs[1].set_xlabel("θ' (rad/s)")
        axs[1].set_ylabel("μ")
        axs[1].grid(True)
        axs[1].legend()

        # Fuerza
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

# =======================
# Crear controlador con conjuntos tipo teoría (NG y PG trapezoidales)
# =======================
def crear_controlador():
    theta_sets = [
        FuzzySet("NG", -91, -91, -90, -45),    # trapezoidal: plano y bajada
        FuzzySet("NP", -90, -45, 0),           # triángulo
        FuzzySet("Z", -45, 0, 45),             # triángulo
        FuzzySet("PP", 0, 45, 90),             # triángulo
        FuzzySet("PG", 45, 90, 91, 91),        # trapezoidal: subida y plano
    ]

    theta_dot_sets = [
        FuzzySet("NG", -11, -11, -10, -5),
        FuzzySet("NP", -10, -5, 0),
        FuzzySet("Z", -5, 0, 5),
        FuzzySet("PP", 0, 5, 10),
        FuzzySet("PG", 5, 10, 11, 11),
    ]

    force_sets = [
        FuzzySet("NG", -31, -31, -30, -15),
        FuzzySet("NP", -30, -15, 0),
        FuzzySet("Z", -15, 0, 15),
        FuzzySet("PP", 0, 15, 30),
        FuzzySet("PG", 15, 30, 31, 31),
    ]

    rules_data = [
        ("NG", "NG", "NG"), ("NP", "NG", "NP"), ("Z", "NG", "NP"), ("PP", "NG", "NP"), ("PG", "NG", "Z"),
        ("NG", "NP", "NG"), ("NP", "NP", "NP"), ("Z", "NP", "NP"), ("PP", "NP", "Z"), ("PG", "NP", "PP"),
        ("NG", "Z", "NG"), ("NP", "Z", "NP"), ("Z", "Z", "Z"), ("PP", "Z", "PP"), ("PG", "Z", "PG"),
        ("NG", "PP", "NP"), ("NP", "PP", "Z"), ("Z", "PP", "PP"), ("PP", "PP", "PP"), ("PG", "PP", "PG"),
        ("NG", "PG", "Z"), ("NP", "PG", "PP"), ("Z", "PG", "PP"), ("PP", "PG", "PP"), ("PG", "PG", "PG"),
    ]

    rules = [FuzzyRule(a1, a2, c) for a1, a2, c in rules_data]

    return FuzzyController(
        FuzzyVariable("theta", theta_sets),
        FuzzyVariable("theta_dot", theta_dot_sets),
        FuzzyVariable("force", force_sets),
        rules
    )


# =============
# Ejemplo de uso
# =============
if __name__ == "__main__":
    theta_input = -65
    theta_dot_input = 6
    controller = crear_controlador()
    output_force = controller.infer(theta_input, theta_dot_input)
    print("Fuerza resultante:", output_force)
    controller.graficar_resultado()
