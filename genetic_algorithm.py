import numpy as np
import random
import time
from operators import (
    tournament_selection, roulette_wheel_selection,
    ox_crossover, pmx_crossover,
    swap_mutation, inversion_mutation
)

class GeneticAlgorithm:
    def __init__(self, tsp_problem, population_size=100, mutation_rate=0.05, 
                 crossover_rate=0.8, elite_size=0.1, selection_method='Tournament',
                 crossover_method='OX', mutation_method='Swap'):
        self.tsp = tsp_problem
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(elite_size * population_size))
        
        self.selection_func = self._get_selection_func(selection_method)
        self.crossover_func = self._get_crossover_func(crossover_method)
        self.mutation_func = self._get_mutation_func(mutation_method)
        
        self.population = self._initialize_population()
        self.best_individual = None
        self.best_fitness = float('inf')
        self.history = []
        
    def _initialize_population(self):
        population = []
        nodes = list(range(self.tsp.dimension))
        for _ in range(self.pop_size):
            ind = nodes.copy()
            random.shuffle(ind)
            population.append(ind)
        return population

    def _get_selection_func(self, name):
        if name == 'Tournament': return tournament_selection
        if name == 'Roulette Wheel': return roulette_wheel_selection
        return tournament_selection

    def _get_crossover_func(self, name):
        if name == 'OX': return ox_crossover
        if name == 'PMX': return pmx_crossover
        return ox_crossover

    def _get_mutation_func(self, name):
        if name == 'Swap': return swap_mutation
        if name == 'Inversion': return inversion_mutation
        return swap_mutation

    def evaluate_population(self):
        fitnesses = [self.tsp.get_total_distance(ind) for ind in self.population]
        min_fit = min(fitnesses)
        if min_fit < self.best_fitness:
            self.best_fitness = min_fit
            self.best_individual = self.population[fitnesses.index(min_fit)].copy()
        return fitnesses

    def evolve(self):
        fitnesses = self.evaluate_population()
        self.history.append(self.best_fitness)
        
        # Elitism
        sorted_indices = np.argsort(fitnesses)
        new_population = [self.population[i].copy() for i in sorted_indices[:self.elite_count]]
        
        # Selection
        selected = self.selection_func(self.population, fitnesses)
        
        # Crossover & Mutation
        while len(new_population) < self.pop_size:
            p1, p2 = random.sample(selected, 2)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self.crossover_func(p1, p2)
            else:
                child = p1.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self.mutation_func(child)
                
            new_population.append(child)
            
        self.population = new_population
        return self.best_individual, self.best_fitness
