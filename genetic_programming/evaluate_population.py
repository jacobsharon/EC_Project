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

def evaluate_population(population, X_train, y_train):
    #1. Standardize labels
    labels = list(y_train)

    #2. Iterate through each tree in the population. 
    for i, tree in enumerate(population):
        tree_func = gp.compile(tree, pset=primitive_set)                #Turn the tree into a python compiled function
        scores = [tree_func(**row) for _, row in X_train.iterrows()]    #Store raw scores from tree func for each row in the training data
        complexity = len(tree)

        #3. Try multiple thresholds to find best F1
        thresholds = np.linspace(min(scores), max(scores), 100)
        best_f1 = 0.0
        best_thresh = 0.0

        #4. Save the best F1 and threshold generated during the threshold sweep
        for t in thresholds:
            preds = [1 if s > t else 0 for s in scores]
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        #5. Add F1 and complexity as metadata to the current tree
        tree.fitness.values = (best_f1, complexity)
        tree.threshold = best_thresh 

    return population