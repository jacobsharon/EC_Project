'''File: evaluation.py
Purpose: Optimize threshold and evaluate each tree in a GP population based on F1 score.

From scikit learn documentation: 
The F1 score can be interpreted as a harmonic mean of the precision and recall, 
where an F1 score reaches its best value at 1 and worst at 0. 
The relative contribution of precision and recall to the F1 score are equal. 
The F1 score is especially useful when you need to take both precision and recall into account. 
It is usually more useful than accuracy, especially if you have an uneven class distribution.
'''

import numpy as np
from sklearn.metrics import f1_score
from deap import gp
from genetic_programming.primitives import primitive_set

#
def evaluate_population(population, X_train, y_train):
    #1. initialize arrays to store f1 scores and their thresholds
    f1_scores = []
    thresholds = []

    #2. iterate through each tree in the population
    for tree in population:
        #3. initialize array to store numeric outputs of the model for threshold evaluation
        #4. initialize array to store ckd/notckd labels
        results = []
        labels = []

        #5. take the gp symbolic tree and return a python function (eg: 1 if 175 > 150 else 0 returns = 1)
        # need this so each tree is evaluated
        tree_func = gp.compile(tree, pset=primitive_set)

        #6. iterate through each instance of the training set and call the tree_func for evaluation
        for i, row in X_train.iterrows():
            try:
                result = tree_func(**row.to_dict())     #unpack the tree to into arguments for the tree function
            except Exception:
                result = 0                              #catches any errors not caught in encoding and constraints
            results.append(result)                      #add the final numerical value of the tree to results
            labels.append(1 if y_train.iloc[i] == 'ckd' else 0) #adds the final classification (0 (notckd) or 1 (ckd)) to labels

        #7. Search for the best threshold
        t_values = np.linspace(min(results), max(results), 100)     #sets threshold values in range of [0,100]
        
        #8. initialize array to store score
        scores = []

        #9. iterate through each possible threshold value [0-100]
        for t in t_values:
            #10. initialize an array to store predictive values (0 or 1)
            preds = []
            for r in results:           #if the result is greater than the threshold append 1
                if r > t:
                    preds.append(1)
                else:
                    preds.append(0)     #else 0
            
            #11. compute the f1 score using the real labels and pedicted labels and append to scores
            score = f1_score(labels, preds)
            scores.append(score)

        #12. Returns best values for the current tree
        best_f1 = max(scores)
        best_t = t_values[scores.index(best_f1)]
        f1_scores.append(best_f1)
        thresholds.append(best_t)

    #13. Return the best tree, its threshold value for classification, and its f1 score. 
    best_index = np.argmax(f1_scores)
    return population[best_index], thresholds[best_index], f1_scores[best_index]