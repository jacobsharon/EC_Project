'''
File: evaluate_population.py

The purpose of this file is to calulate the F1 score of the model. 
'''

# Third Party Libraries
from deap import gp
import numpy as np
from sklearn.metrics import f1_score

# Local Application Modules
from genetic_programming.primitives import primitive_set 

####################################################################
# 1. evaluate_population: Assign fitness based on F1 and complexity #
####################################################################
def evaluate_population(population, X_train, y_train):
    """
    Evaluates each individual in the population by compiling the symbolic expression,
    applying it to the training set, and assigning fitness based on the best F1 score
    achieved over a range of thresholds.

    Parameters:
        population (list): List of DEAP individuals (trees) to evaluate.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): True class labels corresponding to X_train.

    Returns:
        list: The evaluated population with updated fitness values and thresholds.
    """
    # 1.1 Convert label Series to list for performance
    labels = list(y_train)

    # 1.2 Iterate through each tree in the population
    for i, tree in enumerate(population):

        # 1.2.1 Compile the tree into an executable Python function
        tree_func = gp.compile(tree, pset=primitive_set)

        # 1.2.2 Evaluate the tree on each training instance and collect raw output scores
        scores = [tree_func(**row) for _, row in X_train.iterrows()]
        complexity = len(tree)  # Tree size used as complexity

        # 1.2.3 Define thresholds across score range and find one that maximizes F1
        thresholds = np.linspace(min(scores), max(scores), 100)
        best_f1 = 0.0
        best_thresh = 0.0

        # 1.2.4 Sweep across thresholds to compute F1 score at each
        for t in thresholds:
            preds = [1 if s > t else 0 for s in scores]
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        # 1.2.5 Assign best F1 and complexity as fitness to the individual
        tree.fitness.values = (best_f1, complexity)

        # 1.2.6 Store best threshold for later use during test-time prediction
        tree.threshold = best_thresh 

    return population