from random import normalvariate

import numpy as np


def fitness(individual):
    x = individual["x"]
    y = individual["y"]
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

def mutate(individual, sigma=0.1):
    return {"x": individual["x"] + normalvariate(0, sigma), "y": individual["y"] + normalvariate(0, sigma)}

def mutate_towards(individual, reference, impact_factor=0.2):
    return {
        "x": individual["x"] + (reference["x"] - individual["x"]) * impact_factor + normalvariate(0, 0.1), 
        "y": individual["y"] + (reference["y"] - individual["y"]) * impact_factor + normalvariate(0, 0.1)
    }
    