import argparse
import matplotlib
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose

fitness_evals = []
def get_custom_fitness(fitness):
    def counting_fitness(state):
        global fitness_evals
        score = fitness.evaluate(state)
        fitness_evals.append( score )
        return score
    return counting_fitness

def get_fitness_eval_curve():
    fitness_eval_x = []
    fitness_eval_y = []
    maximum = 0
    for i in range(len(fitness_evals)):
        if fitness_evals[i] > maximum:
            maximum = fitness_evals[i]
            fitness_eval_x.append(i)
            fitness_eval_y.append(fitness_evals[i])
    return fitness_eval_x, fitness_eval_y

class OptimizationProblem:
    def __init__(self, complexity, fitness_fn):
        self.problem = mlrose.DiscreteOpt(length = complexity, fitness_fn = fitness_fn) 
    
    def solveWithHillClimbing(self):
        return mlrose.random_hill_climb(problem = self.problem, max_attempts = 500, curve = True)
    
    def solveWithSimulatedAnnealing(self):
        return mlrose.simulated_annealing(problem = self.problem, max_attempts = 500, curve = True)
    
    def solveWithGeneticAlg(self):
        return mlrose.genetic_alg(problem = self.problem, max_attempts = 500, curve = True)
    
    def solveWithMimic(self):
        return mlrose.mimic(problem = self.problem, max_attempts = 500, curve = True)
    
def fitnessVsIterations(fitness):
    OP = OptimizationProblem(complexity = 15, fitness_fn = fitness)
    best_state, best_fitness, rhc_curve = OP.solveWithHillClimbing()
    print( 'Best state found by RHC: %s' % best_state )
    print( 'Best fitness found by RHC: %s' % best_fitness )
    best_state, best_fitness, annealing_curve = OP.solveWithSimulatedAnnealing()
    print( 'Best state found by Simulated Annealing: %s' % best_state )
    print( 'Best fitness found by Simulated Annealing: %s' % best_fitness )
    best_state, best_fitness, genetic_curve = OP.solveWithGeneticAlg()
    print( 'Best state found by Genetic Alg: %s' % best_state )
    print( 'Best fitness found by Genetic Alg: %s' % best_fitness )
    best_state, best_fitness, mimic_curve = OP.solveWithMimic()
    print( 'Best state found by MIMIC: %s' % best_state )
    print( 'Best fitness found by MIMIC: %s' % best_fitness )

    plt.plot(rhc_curve)
    plt.plot(annealing_curve)
    plt.plot(genetic_curve)
    plt.plot(mimic_curve)

    plt.legend(['Random Hill Climbing', 'Simulated Annealing', 'Genetic Alg', 'MIMIC'])
    plt.suptitle('Four Peaks: Fitness Vs Iterations')
    plt.ylabel('Fitness Score')
    plt.xlabel('Iterations')

    plt.show()

def fitnessVsEvaluations(fitness):
    global fitness_evals
    OP = OptimizationProblem(complexity = 15, fitness_fn = fitness)
    fitness_evals = []
    OP.solveWithHillClimbing()
    print('hill climb done')
    x, y = get_fitness_eval_curve()
    plt.plot(x, y)
    fitness_evals = []
    OP.solveWithSimulatedAnnealing()
    print('annealing done')
    x, y = get_fitness_eval_curve()
    plt.plot(x, y)
    fitness_evals = []
    OP.solveWithGeneticAlg()
    print('genetic done')
    x, y = get_fitness_eval_curve()
    plt.plot(x, y)
    fitness_evals = []
    OP.solveWithMimic()
    print('mimic done')
    x, y = get_fitness_eval_curve()
    plt.plot(x, y)

    plt.legend(['Random Hill Climbing', 'Simulated Annealing', 'Genetic Alg', 'MIMIC'])
    plt.suptitle('Four Peaks: Fitness Vs Fitness Function Evaluations')
    plt.ylabel('Fitness Score')
    plt.xlabel('Evaluations')

    plt.show()

def maxIterationsVsProblemComplexity():
    pass

def evaluationsVsProblemComplexity():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--problem', action = 'store', dest = 'problem', required = True )
    args = parser.parse_args()

    if args.problem == 'onemax':
        fitness = mlrose.OneMax()
    elif args.problem == 'four_peaks':
        fitness = mlrose.FourPeaks()
    elif args.problem == 'kcolor':
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        fitness = mlrose.MaxKColor(edges)
        # problem = mlrose.DiscreteOpt(length = 5, fitness_fn = fitness)
    else:
        raise RuntimeError("Invalid problem argument")

    fitnessVsIterations(fitness)

    custom_fitness_function = get_custom_fitness(fitness)
    custom_fitness = mlrose.CustomFitness(custom_fitness_function)

    fitnessVsEvaluations(custom_fitness)