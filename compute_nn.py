import argparse
import mlrose_hiive as mlrose
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

class NNTrainer:
    def __init__(self):
        data = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2, random_state = 3)
        
        # Normalize feature data
        scaler = MinMaxScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        # One hot encode target values
        one_hot = OneHotEncoder()
        self.y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
        self.y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
    
    def run(self, algorithm):
        assert(algorithm in ['random_hill_climb', 'simulated_annealing', 'genetic_alg'])
        print( 'Optimizing weights with learner: %s' % algorithm )
        # Initialize neural network object and fit object - attempt 2
        nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', 
                                        algorithm = algorithm, 
                                        max_iters = 1000, bias = True, is_classifier = True, 
                                        learning_rate = 0.0001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 100, random_state = 3)
        nn_model2.fit(self.X_train_scaled, self.y_train_hot)
        y_train_pred = nn_model2.predict(self.X_train_scaled)
        y_train_accuracy = accuracy_score(self.y_train_hot, y_train_pred)
        print(y_train_accuracy)
        # Predict labels for test set and assess accuracy
        y_test_pred = nn_model2.predict(self.X_test_scaled)
        y_test_accuracy = accuracy_score(self.y_test_hot, y_test_pred)
        print(y_test_accuracy)
    
    def run_gradient_descent(self):
        print( 'Optimizing weights with gradient_descent' )
        # Initialize neural network object and fit object - attempt 1
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu', 
                                        algorithm ='gradient_descent', 
                                        max_iters = 1000, bias = True, is_classifier = True, 
                                        learning_rate = 0.0001, early_stopping = True, 
                                        clip_max = 5, max_attempts = 100, random_state = 3)
        nn_model1.fit(self.X_train_scaled, self.y_train_hot)
        # Predict labels for train set and assess accuracy
        y_train_pred = nn_model1.predict(self.X_train_scaled)
        y_train_accuracy = accuracy_score(self.y_train_hot, y_train_pred)
        print(y_train_accuracy)
        # Predict labels for test set and assess accuracy
        y_test_pred = nn_model1.predict(self.X_test_scaled)
        y_test_accuracy = accuracy_score(self.y_test_hot, y_test_pred)
        print(y_test_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--learner', action = 'store', dest = 'learner', required = True )
    args = parser.parse_args()
    
    nn = NNTrainer()
    nn.run(args.learner)
    nn.run_gradient_descent()