import numpy as np
import random
import time
import os

import numpy as np

def read_equation(filename):
    with open(filename, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    # Split based on delimiter ---
    if '---' in lines:
        delimiter_index = lines.index('---')
        A_lines = lines[:delimiter_index]
        b_lines = lines[delimiter_index + 1:]
        A = [list(map(float, line.split())) for line in A_lines]
        b = [list(map(float, line.split())) for line in b_lines]
    else:
        # No delimiter, b is zero vector
        A = [list(map(float, line.split())) for line in lines]
        b = [0.0] * len(A)

    return np.array(A), np.array(b)


def sign(x):
    return (x > 0) - (x < 0)

def fitness_function(ant,A,b):
    b_pred = np.dot(A, ant)

    # Calculate the Mean Squared Error (MSE)
    fitness = np.mean((b - b_pred) ** 2)

    return fitness

def create_population(num_monkeys, dimension, bounds):
    population = np.random.uniform(bounds[0], bounds[1], (num_monkeys, dimension))
    return population

def calculate_barycenter(monkey_positions):
    return np.mean(monkey_positions, axis=0)

def climbing_spsa(monkey, A, b, a, c, alpha, gamma, max_iterations=50):
    A_const = 0.1
    for i in range(max_iterations):
        delta = np.random.choice([-1, 1], size=len(monkey))
        f_plus = fitness_function(monkey + c * delta, A, b)
        f_minus = fitness_function(monkey - c * delta, A, b)
        pseudo_gradient = (f_plus - f_minus) / (2 * c * delta + 1e-8)

        monkey -= (a / ((A_const + i + 1) ** alpha)) * pseudo_gradient
        c = max(c / ((i + 1) ** gamma), 1e-6)

    return monkey



def watch_jump(monkey, b_eye, A, b, improvement_threshold=1e-2, max_attempts=10):
    best_fitness = fitness_function(monkey, A, b)

    for i in range(max_iterations):
        # Generate a new position with random jumps within the b_eye range
        new_monkey = [random.uniform(x_i - b_eye, x_i + b_eye) for x_i in monkey]

        new_fitness = fitness_function(new_monkey, A, b)

        if new_fitness < best_fitness - improvement_threshold:
            monkey = new_monkey
            best_fitness = new_fitness
        else:
            break

    return monkey


def somersault_process(monkey_positions, bounds, alpha_range=(-1, 1)):
    """
    Perform somersault movements around the barycenter.
    """
    pivot = calculate_barycenter(monkey_positions)

    for i in range(len(monkey_positions)):
        alpha = random.uniform(*alpha_range)
        monkey_positions[i] = monkey_positions[i] + alpha * (monkey_positions[i] - pivot)

        # Clip each dimension individually to respect its bounds
        monkey_positions[i] = np.array([
            np.clip(monkey_positions[i][j], bounds[j][0], bounds[j][1])
            for j in range(len(bounds))
        ])

    return monkey_positions

def adjust_parameters(iteration, max_iterations, dimension):
    factor = 1
    if dimension <= 4:
        factor = 10
    elif dimension <= 10:
        factor = 50
    elif dimension <= 30:
        factor = 100
    elif dimension <= 50:
        factor = 1000

    a_initial, a_final = 0.1/factor, 0.01/factor  # Learning rate starts high and reduces
    c_initial, c_final = 0.1/factor, 0.01/factor  # Perturbation size for SPSA
    b_eye_initial, b_eye_final = 0.2/factor, 0.02/factor  # Range for random jumps

    decay = iteration / max_iterations  # Proportion of progress (0 to 1)

    a = a_initial * (1 - decay) + a_final * decay
    c = c_initial * (1 - decay) + c_final * decay
    b_eye = b_eye_initial * (1 - decay) + b_eye_final * decay


    return a, c, b_eye

def MA(num_monkeys, dimension, initial_bounds, max_iterations, A, b):
    """
    Main Monkey Algorithm (MA) function.
    """
    bounds = [[initial_bounds[0], initial_bounds[1]] for _ in range(dimension)]
    monkey_population = create_population(num_monkeys, dimension, bounds[0])
    fitness = np.array([fitness_function(monkey, A, b) for monkey in monkey_population])

    best_monkey_idx = np.argmin(fitness)
    best_monkey = monkey_population[best_monkey_idx].copy()
    best_fitness = fitness[best_monkey_idx].copy()

    alpha = 0.6
    gamma = 0.1
    somersaultinterval = [-0.01, 0.01]

    i = 0
    unchanged_steps = 0
    while i < max_iterations and best_fitness > 1e-5:
        a, c, b_eye = adjust_parameters(i, max_iterations, dimension)

        for j in range(num_monkeys):
            monkey = monkey_population[j].copy()
            monkey = np.array([
                np.clip(monkey[j], bounds[j][0], bounds[j][1])
                for j in range(len(bounds))
            ])
            if j != best_monkey_idx:
                monkey = climbing_spsa(monkey, A, b, a, c, alpha, gamma)
                monkey = watch_jump(monkey, b_eye, A, b)
                monkey = climbing_spsa(monkey, A, b, a, c, alpha, gamma)
                monkey_population = somersault_process(monkey_population, bounds, somersaultinterval)
                monkey_population[j] = monkey

        # Update the monkey population
        fitness = np.array([fitness_function(monkey, A, b) for monkey in monkey_population])

        # Update best monkey
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_monkey = monkey_population[current_best_idx].copy()
            best_fitness = fitness[current_best_idx].copy()
            unchanged_steps = 0
        else:
            unchanged_steps += 1

        if unchanged_steps > 50:
            for j in range(num_monkeys):
                if fitness[j] > best_fitness * 10:
                    rand_var = round(np.random.uniform(0, dimension - 1))
                    monkey_population[j] = best_monkey.copy()
                    monkey_population[j][rand_var] = random.uniform(bounds[rand_var][0], bounds[rand_var][1])

        print("Iteration:", i, "Best Fitness:", best_fitness)
        # if i % 100 == 0:
        #     print("Best Monkey:", best_monkey)
        #     print("Bounds:", bounds)
        i += 1

    # print("Iterations:", i)
    # print("Fitness:", format(best_fitness, '.8f'))
    return best_monkey, i , best_fitness


if __name__ == "__main__":
    n = 1
    for i in range(n):
        filename = "exemplu.txt"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, "extracted_matrices", filename)
        A, b = read_equation(full_path)

        # Define the problem's parameters
        num_monkeys = 10
        dimension = len(A[0])
        initial_bounds = [-10,10]
        max_iterations = 1000

        t1 = time.time()
        solution, iterations, best_fitness = MA(num_monkeys, dimension, initial_bounds, max_iterations, A, b)
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