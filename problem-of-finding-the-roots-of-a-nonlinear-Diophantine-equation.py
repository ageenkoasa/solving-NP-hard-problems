"""
1. Определим функцию фитнеса.
2. Сгенерируем начальную популяцию.
3. Реализуем операции селекции, скрещивания, мутации и замещения.
4. Объединим все шаги в основной цикл генетического алгоритма.
"""

import random
import numpy as np

def evaluate_fitness(x1, x2, x3, x4, x5, powers, result):
    try:
        equation = sum(
            np.exp(
                np.log(abs(x1) + 1e-10) * powers[i][0] +
                np.log(abs(x2) + 1e-10) * powers[i][1] +
                np.log(abs(x3) + 1e-10) * powers[i][2] +
                np.log(abs(x4) + 1e-10) * powers[i][3] +
                np.log(abs(x5) + 1e-10) * powers[i][4]
            )
            for i in range(5)
        )
    except (OverflowError, ValueError):
        return float('-inf')
    return -abs(equation - result)

def generate_random_solution():
    x1 = random.randint(-100, 100)
    x2 = random.randint(-100, 100)
    x3 = random.randint(-100, 100)
    x4 = random.randint(-100, 100)
    x5 = random.randint(-100, 100)
    return x1, x2, x3, x4, x5

# Селекция
def tournament_selection(population, fitness_values, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(range(len(population)), tournament_size)
        best_index = max(tournament, key=lambda i: fitness_values[i])
        selected_parents.append(population[best_index])
    return selected_parents

# Скрещивание
def one_point_crossover(parent1, parent2):
    """
    Одноточечное скрещивание
    """
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def multi_point_crossover(parent1, parent2):
    """
    Многоточечное скрещивание
    """
    points = sorted(random.sample(range(1, len(parent1)), 2))
    child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
    child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
    return child1, child2

# Мутация
def mutate_solution(solution, p1, p2):
    mutated_solution = list(solution)
    for i in range(len(mutated_solution)):
        if random.random() < p1:
            mutated_solution[i] += random.randint(-1, 1)
        elif random.random() < p2:
            mutated_solution[i] += random.randint(-1, 1)
    return tuple(mutated_solution)

# Замещение
def replace_population(old_population, new_population, fitness_values_old, fitness_values_new):
    combined_population = old_population + new_population
    combined_fitness = fitness_values_old + fitness_values_new
    sorted_indices = np.argsort(combined_fitness)
    return [combined_population[i] for i in sorted_indices[-len(old_population):]]

# Основной цикл генетического алгоритма
def genetic_algorithm(powers, result, num_generations, population_size, tournament_size, crossover_prob, mutation_prob, p1, p2):
    population = [generate_random_solution() for _ in range(population_size)]
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(num_generations):
        fitness_values = [evaluate_fitness(*individual, powers, result) for individual in population]
        
        if max(fitness_values) > best_fitness:
            best_fitness = max(fitness_values)
            best_solution = population[np.argmax(fitness_values)]
        
        parents = tournament_selection(population, fitness_values, tournament_size)
        offspring = []

        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            if random.random() < crossover_prob:
                if random.random() < 0.5:
                    child1, child2 = one_point_crossover(parent1, parent2)
                else:
                    child1, child2 = multi_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            child1 = mutate_solution(child1, p1, p2)
            child2 = mutate_solution(child2, p1, p2)
            offspring.append(child1)
            offspring.append(child2)
        
        fitness_values_new = [evaluate_fitness(*individual, powers, result) for individual in offspring]
        population = replace_population(population, offspring, fitness_values, fitness_values_new)
    
    return best_solution, best_fitness

# Заданные степени и результаты для трех уравнений
powers1 = [
    [1, 0, 2, 2, 2],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 2],
    [0, 0, 1, 2, 2],
    [1, 2, 2, 1, 2]
]
result1 = -50

powers2 = [
    [1, 1, 2, 2, 2],
    [2, 1, 2, 2, 2],
    [0, 1, 1, 1, 0],
    [1, 2, 2, 0, 2],
    [0, 0, 0, 0, 1]
]
result2 = -50

powers3 = [
    [0, 2, 1, 2, 0],
    [2, 1, 2, 2, 0],
    [1, 2, 1, 1, 0],
    [2, 0, 2, 2, 0],
    [2, 0, 0, 0, 1]
]
result3 = -50

# Параметры генетического алгоритма
num_generations = 100
population_size = 100
tournament_size = 5
crossover_prob = 0.8
mutation_prob = 0.1
p1 = 0.1
p2 = 0.01

best_solution1, best_fitness1 = genetic_algorithm(powers1, result1, num_generations, population_size, tournament_size, crossover_prob, mutation_prob, p1, p2)
print(f"Best solution for first equation: {best_solution1}")
print(f"Best fitness for first equation: {best_fitness1}\n")

best_solution2, best_fitness2 = genetic_algorithm(powers2, result2, num_generations, population_size, tournament_size, crossover_prob, mutation_prob, p1, p2)
print(f"Best solution for second equation: {best_solution2}")
print(f"Best fitness for second equation: {best_fitness2}\n")

best_solution3, best_fitness3 = genetic_algorithm(powers3, result3, num_generations, population_size, tournament_size, crossover_prob, mutation_prob, p1, p2)
print(f"Best solution for third equation: {best_solution3}")
print(f"Best fitness for third equation: {best_fitness3}\n")
