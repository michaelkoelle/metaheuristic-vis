import random
from typing import Dict, List

import numpy as np

from plot import Point
from problem import fitness, mutate


class SimulatedAnnealing:
    
    def __init__(self, num_iter: int, visualizer, initial_temperature: float = 100.0, cooling_rate: float = 0.99, bound: float = 2.5):
        assert initial_temperature > 0, "Initial temperature must be positive"
        assert 0 < cooling_rate < 1, "Cooling rate must be between 0 and 1"
        
        self.num_iter = num_iter
        self.visualizer = visualizer
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.bound = bound
        
        # Initialize solution
        self.current_solution = self.generate_random_solution()
        self.best_solution = self.current_solution.copy()
        self.best_cost = fitness(self.best_solution)

        print(f"Initial solution: {self.current_solution} with cost {self.best_cost}")

    def generate_random_solution(self) -> Dict[str, float]:
        """Generate a random solution within the bounds."""
        return {
            "x": random.uniform(-self.bound, self.bound),
            "y": random.uniform(-self.bound, self.bound)
        }

    def run(self):
        temperature = self.initial_temperature
        current_cost = self.best_cost  # Initialize with the cost of the first solution
        
        for iteration in range(self.num_iter):
            points = []
            
            # Generate new solution and compute its cost
            new_solution = mutate(self.current_solution, sigma=0.45)
            new_solution["x"] = np.clip(new_solution["x"], -self.bound, self.bound)
            new_solution["y"] = np.clip(new_solution["y"], -self.bound, self.bound)
            new_cost = fitness(new_solution)

            # Accept new solution if it's better or based on acceptance probability
            if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
                self.current_solution = new_solution.copy()
                current_cost = new_cost

            # Update the best solution found
            if current_cost < self.best_cost:
                self.best_solution = self.current_solution.copy()
                self.best_cost = current_cost
                print(f"New best solution: {self.best_solution} with cost {self.best_cost}")

            # Visualization
            points.append(Point(x=new_solution["x"], y=new_solution["y"], size=5, style="wo"))
            points.append(Point(x=self.best_solution["x"], y=self.best_solution["y"], size=12, style="rX"))
            points.append(Point(x=self.current_solution["x"], y=self.current_solution["y"], size=8, style="co"))
            self.visualizer.capture_frame(points)

            # Cool down the temperature
            temperature *= self.cooling_rate

            # Optionally, you could add a break if temperature goes too low
            if temperature < 1e-10:  # A very small threshold
                print("Stopping early due to very low temperature")
                break

