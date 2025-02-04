import random
from typing import List

import numpy as np

from plot import Point
from problem import fitness, mutate


def lerp(a, b, t):
    return a + t * (b - a)


class ParticleSwarmOptimization:

    def __init__(self, num_particles, num_iter, visualizer, w_max=0.9, w_min=0.4, cog_weight=2.0, social_weight=2.0):
        self.num_particles = num_particles
        self.num_iter = num_iter
        self.visualizer = visualizer
        self.w_max = w_max
        self.w_min = w_min
        self.w = self.w_max  # Start with the maximum inertia weight
        self.cog_weight = cog_weight
        self.social_weight = social_weight
        self.bound = 5.12  # Standard bound for the Rastrigin function
        self.v_max = (self.bound) / 32 # Maximum velocity
        self.v_min = -self.v_max  # Minimum velocity
        self.t = 0.5

        # Initialize particles within bounds
        self.particles = [
            {"x": random.uniform(-self.bound, self.bound),
             "y": random.uniform(-self.bound, self.bound)}
            for _ in range(num_particles)
        ]

        # Initialize velocities within velocity bounds
        self.velocities = [
            {"x": random.uniform(self.v_min, self.v_max),
             "y": random.uniform(self.v_min, self.v_max)}
            for _ in range(num_particles)
        ]

        self.best_local_positions = self.particles.copy()
        self.best_local_costs = [fitness(p) for p in self.particles]
        self.best_global_position = self.best_local_positions[self.best_local_costs.index(min(self.best_local_costs))]
        self.best_global_cost = fitness(self.best_global_position)

        print(f"Initial solution: {self.best_global_position} with cost {self.best_global_cost}")

    def run(self):
        for iteration in range(self.num_iter):
            # Linearly decrease inertia weight
            self.w = self.w_max - ((self.w_max - self.w_min) * (iteration / self.num_iter))
            positions = []
            velocities = []
            for i, particle in enumerate(self.particles):
                # Generate separate random numbers for each dimension
                r1_x, r2_x = random.uniform(0, 1), random.uniform(0, 1)
                r1_y, r2_y = random.uniform(0, 1), random.uniform(0, 1)

                # Update velocity with separate random numbers for each dimension
                new_velocity = {
                }

                new_velocity["x"] = lerp(self.velocities[i]["x"], 
                         self.w * self.velocities[i]["x"] + 
                         self.cog_weight * r1_x * (self.best_local_positions[i]["x"] - particle["x"]) + 
                         self.social_weight * r2_x * (self.best_global_position["x"] - particle["x"]),
                         self.t)

                new_velocity["y"] = lerp(self.velocities[i]["y"], 
                                        self.w * self.velocities[i]["y"] + 
                                        self.cog_weight * r1_y * (self.best_local_positions[i]["y"] - particle["y"]) + 
                                        self.social_weight * r2_y * (self.best_global_position["y"] - particle["y"]),
                                        self.t)

                # Limit velocities
                new_velocity["x"] = max(min(new_velocity["x"], self.v_max), self.v_min)
                new_velocity["y"] = max(min(new_velocity["y"], self.v_max), self.v_min)

                # Update position
                new_position = {
                    "x": particle["x"] + new_velocity["x"],
                    "y": particle["y"] + new_velocity["y"]
                }

                # Boundary checking
                new_position["x"] = max(min(new_position["x"], self.bound), -self.bound)
                new_position["y"] = max(min(new_position["y"], self.bound), -self.bound)

                # Collect positions and velocities for visualization
                positions.append(Point(x=new_position["x"], y=new_position["y"], size=5, style="wo"))
                velocities.append(Point(x=new_velocity["x"], y=new_velocity["y"]))

                # Compute fitness once and store it
                cost = fitness(new_position)

                # Update local and global bests if necessary
                if cost < self.best_local_costs[i]:
                    self.best_local_positions[i] = new_position
                    self.best_local_costs[i] = cost

                    if cost < self.best_global_cost:
                        self.best_global_position = new_position
                        self.best_global_cost = cost
                        print(f"Best solution: {self.best_global_position} with cost {self.best_global_cost}")

                # Update particle's position and velocity
                self.particles[i] = new_position
                self.velocities[i] = new_velocity

            # Add the global best point for visualization
            positions.append(Point(x=self.best_global_position["x"], y=self.best_global_position["y"], size=12, style="rX"))
            velocities.append(Point(x=0, y=0))  # No velocity arrow for the global best point

            # Visualize particles and velocities
            self.visualizer.capture_frame_with_velocities(positions, velocities)



# import random
# from typing import List

# from plot import Point
# from problem import fitness, mutate


# class ParticleSwarmOptimization:

#     def __init__(self, num_particles, num_iter,visualizer, w=0.2, cog_weight=2.0, social_weight=2.0) -> None:
#         self.num_particles = num_particles
#         self.num_iter = num_iter
#         self.visualizer = visualizer
#         self.w = w
#         self.cog_weight = cog_weight
#         self.social_weight = social_weight
#         bound = 2.5
#         self.particles = [{"x": random.uniform(-bound, bound), "y": random.uniform(-bound, bound)} for _ in range(num_particles)]
#         self.velocities = [{"x": 0, "y": 0} for _ in range(num_particles)]
#         self.best_local_positions = self.particles.copy()
#         self.best_local_costs = [fitness(p) for p in self.particles]
#         self.best_global_position = self.best_local_positions[self.best_local_costs.index(min(self.best_local_costs))]
#         self.best_global_cost = fitness(self.best_global_position)
        
#         print(f"Initial solution: {self.best_global_position} with cost {self.best_global_cost}")

#     def run(self):
#         for _ in range(self.num_iter):
#             points = []
#             for i, particle in enumerate(self.particles):
#                 # Generate separate random numbers for each dimension
#                 r1_x, r2_x = random.uniform(0, 1), random.uniform(0, 1)
#                 r1_y, r2_y = random.uniform(0, 1), random.uniform(0, 1)

#                 # Update velocity with separate random numbers for each dimension
#                 new_velocity = {
#                     "x": self.w * self.velocities[i]["x"] +
#                         self.cog_weight * r1_x * (self.best_local_positions[i]["x"] - particle["x"]) +
#                         self.social_weight * r2_x * (self.best_global_position["x"] - particle["x"]),
#                     "y": self.w * self.velocities[i]["y"] +
#                         self.cog_weight * r1_y * (self.best_local_positions[i]["y"] - particle["y"]) +
#                         self.social_weight * r2_y * (self.best_global_position["y"] - particle["y"])
#                 }

#                 # Update position
#                 new_position = {
#                     "x": particle["x"] + new_velocity["x"],
#                     "y": particle["y"] + new_velocity["y"]
#                 }

#                 points.append(Point(x=new_position["x"], y=new_position["y"], size=5, style="wo"))

#                 # Compute fitness once and store it
#                 cost = fitness(new_position)

#                 # Update local and global bests if necessary
#                 if cost < self.best_local_costs[i]:
#                     self.best_local_positions[i] = new_position
#                     self.best_local_costs[i] = cost

#                     if cost < self.best_global_cost:
#                         self.best_global_position = new_position
#                         self.best_global_cost = cost
#                         print(f"Best solution: {self.best_global_position} with cost {self.best_global_cost}")

#                 # Update particle's position and velocity
#                 self.particles[i] = new_position
#                 self.velocities[i] = new_velocity

#             # Visualize particles and global best
#             points.append(Point(x=self.best_global_position["x"], y=self.best_global_position["y"], size=12, style="rX"))
#             self.visualizer.capture_frame(points)
