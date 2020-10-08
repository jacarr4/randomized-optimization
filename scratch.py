import argparse
import matplotlib
import matplotlib.pyplot as plt
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

# note: genetic alg should excel at four peaks,
#       while RHC and annealing are likely to get stuck in the local optima with bigger basins (all 1s or all 0s).
#       annealing and RHC will perform well on onemax because there is only one optimum and it's global (wide basin).
#       MIMIC will perform best on kcolor, per Charles Isbell's paper

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
        # init_state = np.zeros(size)
        init_state = np.array([1,0,0,1,1,0,1,1])
    elif args.problem == 'kcolor':
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        fitness = mlrose.MaxKColor(edges)
        problem = mlrose.DiscreteOpt(length = 5, fitness_fn = fitness)
        init_state = np.array([0, 1, 0, 1, 1])
    else:
        raise RuntimeError("Invalid problem argument")

    best_fitnesses = []
    fitness_curves = []

    for random_seed in range(30):
        if args.learner == 'simulated_annealing':
            # Define decay schedule
            schedule = mlrose.ExpDecay()
            best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 100, max_iters = 1000, curve = True, random_state = random_seed)
        elif args.learner == 'hill_climbing':
            best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, restarts = 100, max_attempts = 100, max_iters = 100, curve = True, random_state = random_seed)
        elif args.learner == 'genetic_alg':
            best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, max_attempts = 100, curve = True)
        elif args.learner == 'mimic':
            best_state, best_fitness, fitness_curve = mlrose.mimic(problem, curve = True, random_state = random_seed)
        else:
            raise ValueError('Invalid learner argument')

        print('The best state found is: ', best_state)
        print('The fitness at the best state is: ', best_fitness)
        best_fitnesses.append(best_fitness)
        fitness_curves.append(fitness_curve)
        print('The curve is: ', fitness_curve)
        print( len( fitness_curve ) )

    print( 'Average best fitness:', np.mean( best_fitnesses ) )
    avg_fitness_curve = [ np.mean( [ c[i] for c in fitness_curves ] ) for i in range(100) ]
    print( 'Average fitness curve:', avg_fitness_curve )
    plt.plot( avg_fitness_curve )
    plt.xlabel( 'Iterations' )
    plt.ylabel( 'Average fitness score' )
    plt.suptitle( args.learner )
    plt.show()