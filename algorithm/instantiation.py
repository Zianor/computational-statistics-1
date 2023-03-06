from deap import base, creator, tools
from algorithm.genetic_algorithm import read_data, create_individual, evaluate, mate, mutate
import random
from tqdm import tqdm

class CausalDiscoveryGA:
    """Class to instantiate the genetic algorithm.
    """

    def __init__(self, IND_SIZE: int=1):
        """Initialize the genetic algorithm.
        """
        self.pop = None
        self.toolbox = None
        self.IND_SIZE = IND_SIZE
        self.X = read_data()
        
    def initialize_env(self):
        """Initialize the GA environment.
        """

        # minimize fitness  
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # 2d list to represent DAG
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # creating population
        toolbox.register("attribute", create_individual, self.X)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attribute, n=self.IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # mate, mutate, select and evaluate functions
        toolbox.register("mate", mate)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)  # TODO: look into selection functions
        toolbox.register("evaluate", evaluate)

        self.toolbox = toolbox


    def start_ga(self):
        """Start the genetic algorithm.
        """

        # create initial population
        pop = self.toolbox.population(n=50)
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in tqdm(range(NGEN)):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = map(self.toolbox.clone, offspring)

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        return pop