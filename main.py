import genetic_programming.creator_setup 
from deap import gp 
from genetic_programming.toolbox import toolbox
from genetic_programming.initialize import initialize_pop
from genetic_programming.evaluate_population import evaluate_population
from classification_model.classification_model import initialize_classification_model

#1. initialize the classification model
folds = initialize_classification_model()

#2. Obtain the training and testing data
for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
    print(f"Fold {i+1} - Train size: {len(y_train)} | Test size: {len(y_test)}")
    print("y_train distribution:", y_train.value_counts().to_dict())

    # 3. Initialize a population for each fold
    initial_pop = initialize_pop(50)

    # 3. Generate offspring
    offspring = []
    while len(offspring) < 50:
        parent1, parent2 = toolbox.tournament_selection(initial_pop, k = 2)
        child1, child2 = map(toolbox.clone, (parent1, parent2))
        toolbox.crossover(child1, child2)
        child1, = toolbox.mutation(child1)
        child2, = toolbox.mutation(child2)
        offspring.extend([child1, child2])
    offspring = offspring[:50]

    # 4. Evaluate fitness for current fold
    evaluate_population(offspring, X_train, y_train)

    #5. Output F1 scores and number of features used
    for child in offspring:
        f1 = child.fitness.values[0]
        terminals = [node for node in child if isinstance(node, gp.Terminal)]
        features = [t for t in terminals if hasattr(t, 'value') and isinstance(t.value, str)]
        print(f"Fitness score: ({f1:.10f},) | Num features: {len(features)}")