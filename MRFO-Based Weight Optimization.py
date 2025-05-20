import numpy as np

# Define number of truss members
NUM_MEMBERS = 10
LENGTHS = np.random.uniform(2.0, 5.0, NUM_MEMBERS)  # Length of each member (m)
DENSITY = 7850  # Steel density in kg/m³

# Objective function: minimize total weight
def truss_weight(area):
    return np.sum(DENSITY * LENGTHS * area)

# Constraint function: ensure areas >= 0.002 m²
def is_valid(area):
    return np.all(area >= 0.002)

# Initialize population
def initialize_population(n_agents, dim, lb, ub):
    return np.random.uniform(lb, ub, (n_agents, dim))

# MRFO Algorithm
def MRFO(obj_func, dim, lb, ub, n_agents=30, max_iter=100):
    X = initialize_population(n_agents, dim, lb, ub)
    best_pos = np.copy(X[0])
    best_score = float("inf")

    for i in range(n_agents):
        if is_valid(X[i]):
            fitness = obj_func(X[i])
            if fitness < best_score:
                best_score = fitness
                best_pos = np.copy(X[i])

    for t in range(max_iter):
        for i in range(n_agents):
            r = np.random.rand()

            if r < 0.5:  # Chain foraging
                alpha = 2 * np.exp(-t / max_iter)
                rand_index = np.random.randint(n_agents)
                X_new = X[i] + alpha * (X[rand_index] - X[i]) * np.random.rand()
            else:  # Cyclone foraging
                beta = 2 * (1 - t / max_iter)
                X_new = X[i] + beta * (best_pos - X[i]) * np.random.rand()

            # Somersault foraging
            somersault_factor = 2
            X_new += somersault_factor * (np.random.rand() * best_pos - np.random.rand() * X[i])

            # Boundary control
            X_new = np.clip(X_new, lb, ub)

            # Apply constraints
            if is_valid(X_new) and obj_func(X_new) < obj_func(X[i]):
                X[i] = X_new
                if obj_func(X_new) < best_score:
                    best_score = obj_func(X_new)
                    best_pos = X_new

        print(f"Iteration {t+1}/{max_iter}, Best Weight = {best_score:.3f} kg")

    return best_pos, best_score

# Run the optimization
if __name__ == "__main__":
    dim = NUM_MEMBERS
    lb = 0.001
    ub = 0.01
    best_area, best_weight = MRFO(truss_weight, dim, lb, ub, n_agents=30, max_iter=100)

    print("\nOptimal Cross-sectional Areas (m²):", best_area)
    print("Minimum Truss Weight (kg):", best_weight)
