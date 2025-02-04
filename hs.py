import random
from typing import Dict, List

import numpy as np

from plot import Point
from problem import fitness, mutate


class HarmonySearch:

    def __init__(self, num_iter: int, visualizer, hms: int = 50, hmcr: float = 0.9, bound: float = 2.5, par: int = 0.3, bw: float = 0.1):
        
        self.num_iter = num_iter
        self.visualizer = visualizer
        self.bound = bound
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        
        # Initialize population
        self.hm = [self.generate_random_solution() for _ in range(self.hms)]
        self.best_solution = self.hm[0].copy()
        self.best_cost = fitness(self.best_solution)

        print(f"Initial solution: {self.best_solution} with cost {self.best_cost}")

    def generate_random_solution(self) -> Dict[str, float]:
        """Generate a random solution within the bounds."""
        return {
            "x": random.uniform(-self.bound, self.bound),
            "y": random.uniform(-self.bound, self.bound)
        }
    
    def run(self):
        current_solution = self.best_solution.copy()

        for iteration in range(self.num_iter):
            points = []

            for key in ["x", "y"]:
                if random.random() < self.hmcr:
                    value = random.choice(self.hm)[key]

                    if random.random() < self.par:
                        value += random.uniform(-self.bw, self.bw)
                        # value = np.clip(value, -self.bound, self.bound)
                else:
                    value = random.uniform(-self.bound, self.bound)
                
                current_solution[key] = value
            
            current_cost = fitness(current_solution)

            if len(self.hm) < self.hms:
                self.hm.append(current_solution.copy())
            else:
                # replace the worst solution if the new solution is better
                worst_index = np.argmax([fitness(s) for s in self.hm])
                if current_cost < fitness(self.hm[worst_index]):
                    self.hm[worst_index] = current_solution.copy()

            if current_cost < self.best_cost:
                self.best_solution = current_solution.copy()
                self.best_cost = current_cost
                print(f"New best solution: {self.best_solution} with cost {self.best_cost}")

            points.extend([Point(x=point["x"], y=point["y"], size=5, style="wo") for point in self.hm])
            points.append(Point(x=self.best_solution["x"], y=self.best_solution["y"], size=12, style="rX"))
            points.append(Point(x=current_solution["x"], y=current_solution["y"], size=8, style="co"))  
            self.visualizer.capture_frame(points)
        return self.best_solution, self.best_cost

           