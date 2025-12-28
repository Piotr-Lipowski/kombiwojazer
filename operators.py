import numpy as np
import random

# --- SELECTION OPERATORS ---

def tournament_selection(population, fitnesses, tournament_size=3):
    selected_indices = []
    pop_size = len(population)
    for _ in range(pop_size):
        candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
        best_idx = candidates[0]
        for idx in candidates[1:]:
            if fitnesses[idx] < fitnesses[best_idx]: # Minimization
                best_idx = idx
        selected_indices.append(best_idx)
    return [population[i].copy() for i in selected_indices]

def roulette_wheel_selection(population, fitnesses):
    # In TSP we minimize distance, so we need to invert fitness
    max_fit = max(fitnesses)
    min_fit = min(fitnesses)
    
    # Adjusted fitness: (max - current) + epsilon
    epsilon = (max_fit - min_fit) * 0.05 + 1e-6
    adj_fitness = [(max_fit - f) + epsilon for f in fitnesses]
    total_fit = sum(adj_fitness)
    
    probs = [f / total_fit for f in adj_fitness]
    
    pop_size = len(population)
    selected_indices = np.random.choice(range(pop_size), size=pop_size, p=probs)
    return [population[i].copy() for i in selected_indices]

# --- CROSSOVER OPERATORS ---

def ox_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    child[start:end+1] = parent1[start:end+1]
    
    current_child_idx = (end + 1) % size
    for i in range(size):
        p2_val = parent2[(end + 1 + i) % size]
        if p2_val not in child[start:end+1]:
            child[current_child_idx] = p2_val
            current_child_idx = (current_child_idx + 1) % size
            
    return child

def pmx_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    child[start:end+1] = parent1[start:end+1]
    
    # Mapping
    for i in range(start, end + 1):
        if parent2[i] not in child:
            val = parent2[i]
            pos = i
            while start <= pos <= end:
                # Find where parent1[pos] is in parent2
                target = parent1[pos]
                pos = parent2.index(target)
            child[pos] = val
            
    # Fill remaining
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    return child

# --- MUTATION OPERATORS ---

def swap_mutation(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def inversion_mutation(individual):
    start, end = sorted(random.sample(range(len(individual)), 2))
    individual[start:end+1] = individual[start:end+1][::-1]
    return individual
