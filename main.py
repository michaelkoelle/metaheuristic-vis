"""Main python file"""

from aco import AntColonyOptimzation
from ga import GeneticAlgorithm
from hs import HarmonySearch
from plot import OptimizationVisualizer
from problem import fitness
from pso import ParticleSwarmOptimization
from sa import SimulatedAnnealing
from tabu import TabuSearch


def main():
    """Main entrypoint of the program"""
    bound = 3.5
    visualizer = OptimizationVisualizer(
        function=fitness,
        x_bounds=(-bound, bound),
        y_bounds=(-bound, bound),
        interval=75
    )
    
    #  ts = TabuSearch(1000,5, visualizer)
    #  best_solution, best_score = ts.run(500)

    # aco = AntColonyOptimzation(20, 3, 20, visualizer)
    # aco.run()

    # pso = ParticleSwarmOptimization(20, 150, visualizer)
    # pso.run()

    # sa = SimulatedAnnealing(500, visualizer)
    # sa.run()
    
    # ga = GeneticAlgorithm(20, visualizer)
    # ga.run()

    hs = HarmonySearch(300, visualizer)
    hs.run()

    # print(f"Best solution: {best_solution} with cost {best_score}")

    visualizer.create_gif()


if __name__ == "__main__":
    main()
