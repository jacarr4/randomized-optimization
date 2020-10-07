import argparse
import mlrose_hiive as mlrose
import numpy as np

# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    
    # Initialize counter
    fitness = 0
    
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):
                
                # If no attacks, then increment counter
                fitness += 1

    return fitness

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--learner', action = 'store', dest = 'learner', required = True )
    parser.add_argument( '--problem', action = 'store', dest = 'problem', required = True )
    parser.add_argument( '--size', action = 'store', dest = 'size', required = False, default = 8)
    args = parser.parse_args()

    size = int(args.size)

    if args.problem == 'queens':
        # fitness = mlrose.Queens()
        fitness = mlrose.CustomFitness(queens_max)
        problem = mlrose.DiscreteOpt(length = size, fitness_fn = fitness, maximize = True, max_val = size)
        # Define initial state
        init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    elif args.problem == 'onemax':
        fitness = mlrose.OneMax()
        problem = mlrose.DiscreteOpt(length = size, fitness_fn = fitness, maximize = True)
        # init_state = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        init_state = np.zeros(size)
    elif args.problem == 'four_peaks':
        fitness = mlrose.FourPeaks(t_pct = 0.15)
        problem = mlrose.DiscreteOpt(length = size, fitness_fn = fitness)
        # init_state = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        init_state = np.zeros(size)
    elif args.problem == 'kcolor':
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        fitness = mlrose.MaxKColor(edges)
        problem = mlrose.DiscreteOpt(length = 5, fitness_fn = fitness)
        init_state = np.array([0, 1, 0, 1, 1])
    else:
        raise RuntimeError("Invalid problem argument")

    if args.learner == 'simulated_annealing':
        # Define decay schedule
        schedule = mlrose.ExpDecay()
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 100, max_iters = 1000, init_state = init_state, random_state = 1, curve = True)
    elif args.learner == 'hill_climbing':
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, restarts = 100, max_attempts = 100, max_iters = 100, init_state = init_state, random_state = 1, curve = True)
    elif args.learner == 'genetic_alg':
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, max_attempts = 100, curve = True)
    elif args.learner == 'mimic':
        best_state, best_fitness, fitness_curve = mlrose.mimic(problem)
    else:
        raise ValueError('Invalid learner argument')

    print('The best state found is: ', best_state)
    print('The fitness at the best state is: ', best_fitness)
