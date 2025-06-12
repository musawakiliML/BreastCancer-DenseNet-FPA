import numpy as np
import math
from sklearn.feature_selection import mutual_info_classif

class FPAFeatureSelector:
    """Implements Flower Pollination Algorithm for feature selection"""
    
    def __init__(self, n_features, n_population=20, p=0.8, 
                 alpha=0.1, gamma=0.1, max_iter=100):
        """
        Initialize FPA selector
        
        Args:
            n_features (int): Total number of features
            n_population (int): Number of flowers/pollinators
            p (float): Switch probability between global and local pollination
            alpha (float): Scaling factor
            gamma (float): Levy flight parameter
            max_iter (int): Maximum iterations
        """
        self.n_features = n_features
        self.n_population = n_population
        self.p = p
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.best_solution = None
        self.best_fitness = -np.inf
        
    def levy_flight(self, size):
        """Generate step sizes using Levy flight"""
        beta = 1.5  # Levy distribution parameter
        sigma = (math.gamma(1+beta) * np.sin(np.pi*beta/2)) / (math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)) ** (1/beta)
        
        u = np.random.normal(0, sigma, size=size)
        v = np.random.normal(0, 1, size=size)
        step = u / (np.abs(v) ** (1/beta))
        
        return step
    
    def initialize_population(self):
        """Initialize flower population"""
        return np.random.rand(self.n_population, self.n_features)
    
    def evaluate_fitness(self, population, X, y):
        """Evaluate fitness using mutual information"""
        fitness = []
        for solution in population:
            selected_indices = solution > 0.5  # Threshold for feature selection
            if np.sum(selected_indices) == 0:
                fitness.append(0)
                continue
                
            X_selected = X[:, selected_indices]
            mi = mutual_info_classif(X_selected, y, random_state=42)
            avg_mi = np.mean(mi)
            fitness.append(avg_mi)
            
        return np.array(fitness)
    
    def optimize(self, X, y):
        """Run FPA optimization"""
        population = self.initialize_population()
        fitness = self.evaluate_fitness(population, X, y)
        
        best_idx = np.argmax(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        
        for _ in range(self.max_iter):
            new_population = population.copy()
            
            for i in range(self.n_population):
                if np.random.rand() < self.p:
                    # Global pollination
                    L = self.levy_flight(1)
                    new_population[i] += self.alpha * L * (self.best_solution - population[i])
                else:
                    # Local pollination
                    j, k = np.random.choice(self.n_population, 2, replace=False)
                    epsilon = np.random.rand()
                    new_population[i] += epsilon * (population[j] - population[k])
                    
                # Apply simple bounds
                new_population[i] = np.clip(new_population[i], 0, 1)
                
            new_fitness = self.evaluate_fitness(new_population, X, y)
            
            # Update population if fitness improves
            improved = new_fitness > fitness
            population[improved] = new_population[improved]
            fitness[improved] = new_fitness[improved]
            
            # Update best solution
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self.best_fitness:
                self.best_solution = population[current_best_idx].copy()
                self.best_fitness = fitness[current_best_idx]
                
        return self.best_solution
    
    def get_selected_features(self, threshold=0.5):
        """Get selected feature indices based on threshold"""
        if self.best_solution is None:
            raise ValueError("Optimization has not been run yet")
        return np.where(self.best_solution > threshold)[0]
