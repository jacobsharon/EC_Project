from deap import tools, gp
from genetic_programming.toolbox import toolbox
from genetic_programming.initialize import initialize_pop
from genetic_programming.evaluate_population import evaluate_population
from classification_model.classification_model import initialize_classification_model
from settings import POPULATION, GENERATIONS
import matplotlib.pyplot as plt
import numpy as np
from deap.tools.emo import sortNondominated
import os
import graphviz
from results.visualize_tree import save_best_tree_as_image

#1. Initialize the classification model
folds = initialize_classification_model()
X_train, X_test, y_train, y_test = folds[0]

#2. Initialize population and array to store best pareto fronts
population = initialize_pop(POPULATION)
best_pareto_fronts = []

#3. Evaluate fitness of population (f1, complexity) (resued code for checking fitness after operators)
invalid_ind = [ind for ind in population if not ind.fitness.valid]
if invalid_ind:
    evaluate_population(invalid_ind, X_train, y_train)

#4 Run evolutionary loop
for gen in range(GENERATIONS):
    
    # Initialize an array to store the offspring
    offspring = []

    # Loop until an equal number of offspring are produced
    while len(offspring) < POPULATION:

        #Selection: Fitness based Tournemanet Seelction
        parent1, parent2 = toolbox.tournament_selection(population, k=2, fit_attr = "fitness")
        
        #Clone the parents to avoid making changes to originals
        child1, child2 = map(toolbox.clone, (parent1, parent2))
        
        #Delete the clone fitness values so they can take on new fitness later
        del child1.fitness.values
        del child2.fitness.values

        # Crossover
        toolbox.crossover(child1, child2)

        # Mutation: mutUniform returns a tuple
        child1, = toolbox.mutation(child1)
        child2, = toolbox.mutation(child2)

        # Append offspring to the offspring array
        offspring.extend([child1, child2])

    # Limit to just population in case of more operations
    offspring = offspring[:POPULATION]

    # Evaluate all again prior to NSGA-II (DEBUGGING)
    combined = population + offspring
    invalid_ind = [ind for ind in combined if not ind.fitness.valid]
    if invalid_ind:
        evaluate_population(invalid_ind, X_train, y_train)

    # Apply NSGA-II to select the best 50 from the combined initial population and offspring (elitist)
    population = tools.selNSGA2(population + offspring, k=POPULATION)

    # Evaluate the population found by NSGA-II and find the best pareto front (first_front_only = True)
    front = tools.sortNondominated(population, k=len(population), first_front_only=True)[0]
    
    # Append the best pareto front from that population to the array of pareto fronts
    best_pareto_fronts.append(front)

    #This outputs the gen, fitness, and complexity for the first three solutions/gen. Mainly a ticker so I can see progress in terminal. 
    print(f"\nGeneration {gen+1} completed")
    for child in population[:3]:
        f1, complexity = child.fitness.values
        terminals = [n for n in child if isinstance(n, gp.Terminal)]
        features = [t for t in terminals if hasattr(t, 'value') and isinstance(t.value, str)]
        print(f"Fitness: {f1:.4f} | Features: {int(complexity)}")

#5. Plot the progression of the best pareto fronts over each generation

# Colors and styles
highlight_color = 'red'
highlight_size = 40
normal_color = 'blue'
normal_size = 40

# Plot selected generations to see progress over time without cluttering graph
for i in [0, 9, 19, 29, 39, 49]:
    generation = best_pareto_fronts[i]
    f1_scores = [ind.fitness.values[0] for ind in generation]
    complexities = [ind.fitness.values[1] for ind in generation]
    
    if i == 49:  # Last generation (Gen 50)
        plt.scatter(complexities, f1_scores, 
                    label=f"Generation {i+1} (Final)", 
                    color=highlight_color, 
                    s=highlight_size, 
                    edgecolor='black', 
                    linewidths=1.2)
    else:
        plt.scatter(complexities, f1_scores, 
                    label=f"Generation {i+1}", 
                    s=normal_size, 
                    alpha=0.6)

# Plt settings
plt.xlabel("Complexity: Tree Size")
plt.ylabel("F1 Score")
plt.title("Best Pareto at Selected Generations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/Best Pareto at Selected Generations.png")
plt.show()

# Visualize the trees from the pareto front in the 50th gen and save to tree_vsualizations
for i, tree in enumerate(best_pareto_fronts[49]):
    save_best_tree_as_image(tree, f"results/tree_visualizations/Gen_50_Tree{i+1}")

# Write each tree's function expression and its fitness metrics to a TXT file
with open("results/Gen_50_Tree_Functions.txt", "w") as f:
    for i, tree in enumerate(best_pareto_fronts[49]):
        f1_score, complexity = tree.fitness.values
        f.write(f"Individual {i + 1}:\n")
        f.write(f"Function: {str(tree)}\n")
        f.write(f"F1 Score: {f1_score:.4f}, Complexity (Tree Size): {int(complexity)}\n")
        f.write("-" * 40 + "\n")