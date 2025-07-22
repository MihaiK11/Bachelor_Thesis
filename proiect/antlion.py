import numpy as np
import random
import time
import os

def read_equation(filename):
    with open(filename, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    if '---' in lines:
        delimiter_index = lines.index('---')
        A_lines = lines[:delimiter_index]
        b_lines = lines[delimiter_index + 1:]
        A = [list(map(float, line.split())) for line in A_lines]
        b = [list(map(float, line.split())) for line in b_lines]
    else:
        A = [list(map(float, line.split())) for line in lines]
        b = [0.0] * len(A)

    return np.array(A), np.array(b)


def fitness_function(ant,A,b):
    b_pred = np.dot(A, ant)

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((b - b_pred) ** 2)

    return mse

def initialize_population(num_ants, dimension, bounds):
    population = np.random.uniform(bounds[0], bounds[1], (num_ants, dimension))
    return population

def roulette_wheel_selection(fitness):
    inverted_fitness = 1 / (fitness + 1e-10)  # Avoid division by zero
    total_fitness = np.sum(inverted_fitness)
    selection_probs = inverted_fitness / total_fitness
    return np.random.choice(np.arange(len(fitness)), p=selection_probs)

def value_for_I(t,T_max):
    t += 1
    w = 0
    if 0.95*T_max <= t <= T_max:
        w = 6
    elif 0.9*T_max <= t < 0.95*T_max:
        w = 5
    elif 0.75*T_max <= t < 0.9*T_max:
        w = 4
    elif 0.5*T_max <= t < 0.75*T_max:
        w = 3
    elif 0.1*T_max <= t < 0.5*T_max:
        w = 2
    else:
        w = 1

    I = (10**w) * (t/T_max)
    return I

def update_boundaries(selected_antlion, elite_antlion):
    Lb = np.minimum(selected_antlion, elite_antlion) - 0.01 * np.abs(np.minimum(selected_antlion))
    Ub = np.maximum(selected_antlion, elite_antlion) + 0.01 * np.abs(np.maximum(selected_antlion))

    return Lb, Ub

def random_walk(iterations):
    X = np.zeros(iterations+1)
    X[0] = 0

    for i in range(1, iterations+1):
        rand = random.random()
        if rand < 0.5:
            X[i] = X[i - 1] + 1
        else:
            X[i] = X[i - 1] - 1

    return X[1:]

def normalize(X,c_i,d_i,count):
    a = np.min(X)
    b = np.max(X)

    X[count] = (((X[count] - a) * (d_i - c_i)) / (b - a)) + c_i

    return X

def update_boundaries(selected_antlion, elite_antlion):
    Lb = min(elite_antlion) - 0.01 * np.abs(min(selected_antlion))
    Ub = max(elite_antlion) + 0.01 * np.abs(max(selected_antlion))
    return Lb, Ub

def ALO(num_ants, num_antlions, dimension, initial_bounds, max_iter,A,b):
    bounds = [[initial_bounds[0], initial_bounds[1]] for _ in range(dimension)]
    # Initialize the population
    ants = initialize_population(num_ants, dimension, bounds[0])
    antlions = initialize_population(num_antlions, dimension, bounds[0])

    #Initialize the fitness of the ants and antlions
    ants_fitness = np.array([fitness_function(ant,A,b) for ant in ants])
    antlions_fitness = np.array([fitness_function(antlion,A,b) for antlion in antlions])

    elite_antilon_idx = np.argmin(antlions_fitness)
    elite_antlion = antlions[elite_antilon_idx].copy()
    elite_antlion_fitness = antlions_fitness[elite_antilon_idx].copy()

    step = 0
    unchanged_steps = 0
    while step < max_iter and elite_antlion_fitness > 1e-5:
        minimum_c_i = np.zeros((1, ants.shape[1]))
        maximum_d_i = np.zeros((1, ants.shape[1]))
        minimum_c_e = np.zeros((1, ants.shape[1]))
        maximum_d_e = np.zeros((1, ants.shape[1]))
        I = value_for_I(step, max_iter)

        # Update the position of the ants
        for i in range(num_ants):
            ant = ants[i]
            selected_antlion_idx = roulette_wheel_selection(fitness=antlions_fitness)
            sorted_antlions = antlions[antlions[:, -1].argsort()]

            # Loop through each dimension in the ant
            for j in range(ant.shape[0]):
                min_c = sorted_antlions[0, j] / I
                max_d = sorted_antlions[-1, j] / I

                minimum_c_i[0, j] = min_c
                maximum_d_i[0, j] = max_d
                minimum_c_e[0, j] = min_c
                maximum_d_e[0, j] = max_d

                # Generate random numbers in one call
                rand1 = np.random.random()
                rand2 = np.random.random()

                if rand1 < 0.5:
                    minimum_c_i[0, j] += antlions[selected_antlion_idx, j]
                    minimum_c_e[0, j] += elite_antlion[j]
                else:
                    minimum_c_i[0, j] = -min_c + antlions[selected_antlion_idx, j]
                    minimum_c_e[0, j] = -min_c + elite_antlion[j]

                if rand2 >= 0.5:
                    maximum_d_i[0, j] += antlions[selected_antlion_idx, j]
                    maximum_d_e[0, j] += elite_antlion[j]
                else:
                    maximum_d_i[0, j] = -max_d + antlions[selected_antlion_idx, j]
                    maximum_d_e[0, j] = -max_d + elite_antlion[j]

                # Update the ant's position with optimized random walk and normalization
                x_random_walk = normalize(random_walk(max_iter), minimum_c_i[0, j], maximum_d_i[0, j], count=step)
                e_random_walk = normalize(random_walk(max_iter), minimum_c_e[0, j], maximum_d_e[0, j], count=step)

                # Store the updated position for the ant
                instance = (x_random_walk[step] + e_random_walk[step]) / 2
                ants[i, j] = np.clip(instance, bounds[j][0], bounds[j][1])

            # Calculate fitness for the ant
            ants_fitness[i] = fitness_function(ants[i], A, b)
        # Calculate the ant's fitness
        ants_fitness = np.array([fitness_function(ant, A, b) for ant in ants])

        for i in range(num_antlions):
            if ants_fitness[i] < antlions_fitness[i]:
                antlions[i] = ants[i]
                antlions_fitness[i] = ants_fitness[i]

        # Update the best antlion
        best_antilon_idx = np.argmin(antlions_fitness)
        if antlions_fitness[best_antilon_idx] < elite_antlion_fitness:
            elite_antlion = antlions[best_antilon_idx].copy()
            elite_antlion_fitness = antlions_fitness[best_antilon_idx].copy()
            unchanged_steps = 0
        else:
            unchanged_steps += 1

        if unchanged_steps > 30:
            for j in range(num_antlions):
                if antlions_fitness[j] > elite_antlion_fitness * 2:
                    antlions[j] = np.array([
                        random.uniform(bounds[j][0], bounds[j][1]) for j in range(dimension)
                    ])
        print("Iteration = ", step, " f(x) = ", elite_antlion_fitness)
        step += 1

    return elite_antlion, step, elite_antlion_fitness

if __name__ == '__main__':
    n = 1
    for _ in range(n):
        filename = "exemplu.txt"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, "extracted_matrices", filename)
        A, b = read_equation(full_path)

        # Define the problem's parameters
        num_ants = 30
        num_antlions = int(0.7 * num_ants)
        dimension = len(A[0])
        bounds = [-10, 10]
        max_iterations = 550

        # Run the Ant Lion Optimization algorithm
        t1 = time.time()
        solution, iterations, best_fitness = ALO(num_ants, num_antlions, dimension, bounds, max_iterations, A, b)
        t2 = time.time()

        print("Time taken:", t2 - t1)
        print("Iterations:", iterations)
        print("Fitness:", format(best_fitness, '.8f'))
        print("Solution:", solution)

        # Save results to a file
        # with open(f"results_{filename}", "a") as file:
        #     file.write("Solution x:\n")
        #     file.write(" ".join(f"{value}" for value in solution) + "\n")
        #     file.write(f"Iterations: {iterations}\n")
        #     file.write(f"Fitness: {format(best_fitness, '.8f')}\n")
        #     file.write(f"Time taken: {t2 - t1}\n\n")


