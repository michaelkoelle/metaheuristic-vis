import random
from typing import List

from plot import Point
from problem import fitness, mutate


class TabuSearch:
    best_solution = None
    best_cost = None
    tabu_list = []
    tabu_list_size = None
    neighborhood_size = None
    current_solution = None
    rounding_precision = 1

    def __init__(self, tabu_list_size, neighborhood_size, visualizer):
        self.visualizer = visualizer
        self.tabu_list_size = tabu_list_size
        self.neighborhood_size = neighborhood_size
        bound = 2.5
        self.current_solution =  {"x": random.uniform(-bound, bound), "y": random.uniform(-bound, bound)}
        self.best_solution = self.current_solution 
        self.best_cost = fitness(self.best_solution)

        print(f"Initial solution: {self.current_solution} with cost {fitness(self.current_solution)}")

    def round_sol(self, sol):
        return (round(sol["x"], self.rounding_precision), round(sol["y"], self.rounding_precision))

    def run(self, iterations):
        for _ in range(iterations):
            #print(f"Best solution: {self.best_solution} with cost {self.best_cost}") 
            #print(f"Current solution: {self.current_solution} with cost {fitness(self.current_solution)}")
            #print(f"Tabu list: {self.tabu_list}")
            #print("")          
            points:List[Point] = []
            neighborhood = []
            for _ in range(self.neighborhood_size):
                neighbor = mutate(self.current_solution)
                neighborhood.append(neighbor)
                points.append(Point(x=neighbor["x"], y=neighbor["y"], size=5, style="wo"))


            # print(f"Neighborhood: {neighborhood}")

            filtered_neighborhood = list(filter(lambda x: self.round_sol(x) not in self.tabu_list, neighborhood))
            
            filtered_neighborhood.sort(key=lambda x: fitness(x))

            self.current_solution = filtered_neighborhood[0].copy() if len(filtered_neighborhood) > 0 else self.current_solution.copy()
            current_cost = fitness(self.current_solution)

            if  current_cost <= self.best_cost:
                self.best_solution = self.current_solution.copy()
                self.best_cost = current_cost
                
            points.append(Point(x=self.best_solution["x"], y=self.best_solution["y"], size=12, style="rX"))
            points.append(Point(x=self.current_solution["x"], y=self.current_solution["y"],size=8, style="co"))
            
            rounded_current_solution = (round(self.current_solution["x"], self.rounding_precision), round(self.current_solution["y"], self.rounding_precision))
            self.tabu_list.append(rounded_current_solution)

            if len(self.tabu_list) > self.tabu_list_size:
                self.tabu_list.pop(0)
            
            self.visualizer.capture_frame(points)
        return self.best_solution, self.best_cost


            



