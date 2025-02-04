import random
from typing import List

import numpy as np

from plot import Point
from problem import fitness, mutate, mutate_towards


class AntColonyOptimzation:

    def __init__(self, num_points, num_ants, num_iterations, visualizer, evaporation_rate=0.5, alpha=1, beta=1):
        self.visualizer = visualizer
        self.num_ants = num_ants
        self.num_iterations = num_iterations
       
        bound = 2.5
        self.best_solution =  {"x": random.uniform(-bound, bound), "y": random.uniform(-bound, bound)}
        self.best_cost = fitness(self.best_solution)
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.points = [{"x": random.uniform(-bound, bound), "y": random.uniform(-bound, bound)} for _ in range(num_points)]
        self.pheromone_matrix = np.ones((num_points, num_points))

        print(f"Initial solution: {self.best_solution} with cost {fitness(self.best_solution)}")

    def run(self):
        for _ in range(self.num_iterations):
            
            new_solutions = []
            for _ in range(self.num_ants):
                new_solution = random.choice(self.points).copy()
                ant_path = []

                # ant is walking
                for (i, _) in enumerate(self.points):
                    plot_points: List[Point] = []
                    scores = [fitness(point) for point in self.points]
                    min_score = min(scores)
                    pos_scores = [score - min_score + 1e-5 for score in scores]
                    probabilities = np.power(self.pheromone_matrix[i], self.alpha) * np.power(pos_scores, self.beta)
                    probabilities_sum = np.sum(probabilities)

                    if probabilities_sum > 0:
                        probabilities /= probabilities_sum
                    else:
                        probabilities = np.ones(len(self.points)) / len(self.points)

                    
                    selected_index = np.random.choice(range(len(self.points)), p=probabilities)
                    target = self.points[selected_index].copy()
                    new_solution = mutate_towards(new_solution, target)
                    ant_path.append(new_solution)

                    plot_points.append(Point(x=new_solution["x"], y=new_solution["y"], size=8, style="co"))
                    plot_points.extend([Point(x=point["x"], y=point["y"], size=4, style="co") for point in ant_path])
                    plot_points.extend([Point(x=point["x"], y=point["y"], size=5, style="wo") for point in self.points])
                    plot_points.extend([Point(x=point["x"], y=point["y"], size=5, style="mx") for point in new_solutions])

                    plot_points.append(Point(x=self.best_solution["x"], y=self.best_solution["y"], size=12, style="rX"))
            
                    self.visualizer.capture_frame(plot_points)
                
                new_solutions.append(new_solution)
                
                scores = [fitness(solution) for solution in new_solutions]
                best_index = np.argmin(scores)

                if scores[best_index] < self.best_cost:
                    self.best_solution = new_solutions[best_index].copy()
                    self.best_cost = scores[best_index]
                    print(f"New best solution: {self.best_solution} with cost {self.best_cost}")


            
            # pheromone evaporation
            self.pheromone_matrix *= (1 - self.evaporation_rate)

            # pheromone deposition
            for i, solution in enumerate(new_solutions):
                scores = fitness(solution)
                self.pheromone_matrix[i] += 1 / (scores + 1e-5)
            
            