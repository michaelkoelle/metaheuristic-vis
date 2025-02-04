import random
from typing import Dict, List

import numpy as np

from plot import Point
from problem import fitness, mutate


class GeneticAlgorithm:

    def __init__(self, num_iter: int, visualizer, population_size: int = 20, mutation_rate: float = 0.3, bound: float = 2.5, retain: int = 5):
        assert population_size > 0, "Population size must be positive"
        assert 0 < mutation_rate < 1, "Mutation rate must be between 0 and 1"
        
        self.num_iter = num_iter
        self.visualizer = visualizer
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.bound = bound
        self.retain = retain
        
        # Initialize population
        self.population = [self.generate_random_solution() for _ in range(population_size)]
        self.best_solution = self.population[0].copy()
        self.best_cost = fitness(self.best_solution)

        print(f"Initial solution: {self.best_solution} with cost {self.best_cost}")

    def generate_random_solution(self) -> Dict[str, float]:
        """Generate a random solution within the bounds."""
        return {
            "x": random.uniform(-self.bound, self.bound),
            "y": random.uniform(-self.bound, self.bound)
        }
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Perform crossover between two parents."""
        child = {}
        for key in parent1.keys():
            child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        return child
    
    def run(self):
        for iteration in range(self.num_iter):
            points = []

            fitness_values = [fitness(individual) for individual in self.population]

            # Sort population based on fitness
            sorted_population = [x for _, x in sorted(zip(fitness_values, self.population))]

            # Retain the best individuals
            elite = sorted_population[0]
            best_solutions = sorted_population[:self.retain]

            points.append(Point(x=self.best_solution["x"], y=self.best_solution["y"], size=12, style="rX"))
            points.extend([Point(x=point["x"], y=point["y"], size=5, style="co") for point in best_solutions])
            self.visualizer.capture_frame(points)

            points = []
            # create new population from best individuals using mutation and crossover
            new_population = []
            for i in range(self.population_size - 1):
                parent1, parent2 = random.choices(best_solutions, k=2)
                child = self.crossover(parent1, parent2)
                child = mutate(child, sigma=self.mutation_rate)
                new_population.append(child)
                points.append(Point(x=child["x"], y=child["y"], size=5, style="wo"))
            
            # Add elite to the new population
            new_population.append(elite)
            
            # Update population
            self.population = new_population
            
            # Update best solution
            for individual in self.population:
                cost = fitness(individual)
                if cost < self.best_cost:
                    self.best_solution = individual.copy()
                    self.best_cost = cost
                    print(f"New best solution: {self.best_solution} with cost {self.best_cost}")
            
            # Visualization
            points.append(Point(x=self.best_solution["x"], y=self.best_solution["y"], size=12, style="rX"))
            self.visualizer.capture_frame(points)
           