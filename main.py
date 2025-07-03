'''
File: main.py

The purpose of this file is to serve as the main script for running the Eolutionary Computing Project (EC_Project)
SHAP analysis is run seperately in its own module due to time requirements. 
To run shap execute "python -m explain.SHAP_analysis" in terminal. 
'''

# Standard library
import os

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from deap import tools, gp

# Local application modules
from genetic_programming.toolbox import toolbox
from genetic_programming.initialize import initialize_pop
from genetic_programming.evaluate_population import evaluate_population
from genetic_programming.primitives import primitive_set
from classification_model.classification_model import initialize_classification_model
from results.visualize_tree import save_best_tree_as_image
from settings import GENERATIONS, POPULATION, MAX_DEPTH

#1. Initialize the classification model
X_train, X_test, y_train, y_test = initialize_classification_model()[0]

#2. Iterate through each depth and population
for depth in MAX_DEPTH:
    for pop in POPULATION:

        #3. Update toolbox max depth before initializing population (clear any decorators first)
        if hasattr(toolbox.crossover, "decorators"):
            toolbox.crossover.decorators.clear()
        if hasattr(toolbox.mutation, "decorators"):
            toolbox.mutation.decorators.clear()
        toolbox.decorate("crossover", gp.staticLimit(key=len, max_value=depth))
        toolbox.decorate("mutation", gp.staticLimit(key=len, max_value=depth))

        #4. Set file path for results
        base_dir = f"results/pop_{pop}/max_depth_{depth}"
        plot_dir = os.path.join(base_dir, "plots")
        tree_dir = os.path.join(base_dir, "tree_visualizations")
        shap_dir = os.path.join(base_dir, "SHAP_analysis")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(tree_dir, exist_ok=True)
        os.makedirs(shap_dir, exist_ok=True)

        #5. Initialize population and array to store best pareto fronts
        population = initialize_pop(pop)
        best_pareto_fronts = []

        #6. Evaluate fitness of population (f1, complexity)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        if invalid_ind:
            evaluate_population(invalid_ind, X_train, y_train)

        #7 Run evolutionary loop
        for gen in range(GENERATIONS):

            # Initialize an array to store the offspring
            offspring = []

            # Loop until an equal number of offspring are produced
            while len(offspring) < pop:

                # Selection: Fitness based Tournament Selection
                parent1, parent2 = toolbox.tournament_selection(population, k=2, fit_attr = "fitness")

                # Clone the parents to avoid making changes to originals
                child1, child2 = map(toolbox.clone, (parent1, parent2))

                # Delete the clone fitness values so they can take on new fitness later
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
            offspring = offspring[:pop]

            # Evaluate all again prior to NSGA-II
            combined = population + offspring
            invalid_ind = [ind for ind in combined if not ind.fitness.valid]
            if invalid_ind:
                evaluate_population(invalid_ind, X_train, y_train)

            # Apply NSGA-II to select the best 50 from the combined initial population and offspring (elitist)
            population = tools.selNSGA2(population + offspring, k=pop)

            # Evaluate the population found by NSGA-II and find the best pareto front (first_front_only = True)
            front = tools.sortNondominated(population, k=len(population), first_front_only=True)[0]

            # Deduplicate solutions by tree structure and fitness
            unique = []
            seen = set()

            for ind in front:
                key = (str(ind), ind.fitness.values)
                if key not in seen:
                    seen.add(key)
                    unique.append(ind)

            front = unique  # Update with only unique individuals

            # Append the best pareto front from that population to the array of pareto fronts
            best_pareto_fronts.append(front)

            # This outputs the gen, fitness, and complexity for the first three solutions/gen. Mainly a ticker so I can see progress in terminal. 
            print(f"\n=== Population {pop} | Depth {depth} | Generation {gen+1} completed ===")
            for child in population[:3]:
                f1, complexity = child.fitness.values
                terminals = [n for n in child if isinstance(n, gp.Terminal)]
                features = [t for t in terminals if hasattr(t, 'value') and isinstance(t.value, str)]
                print(f"Fitness: {f1:.4f} | Features: {int(complexity)}")

        #8. Plot the pareto fronts over generations
        highlight_color = 'red'
        highlight_size = 40
        normal_color = 'blue'
        normal_size = 40

        for i in [0, 9, 19, 29, 39, 49]:
            generation = best_pareto_fronts[i]
            f1_scores = [ind.fitness.values[0] for ind in generation]
            complexities = [ind.fitness.values[1] for ind in generation]

            if i == 49:  # Last generation (Gen 50)
                plt.scatter(complexities, f1_scores, 
                            label=f"Generation {i+1} Pop {pop} Max Depth {depth} (Final)", 
                            color=highlight_color, 
                            s=highlight_size, 
                            edgecolor='black', 
                            linewidths=1.2)
            else:
                plt.scatter(complexities, f1_scores, 
                            label=f"Generation {i+1}", 
                            s=normal_size, 
                            alpha=0.6)

        plt.xlabel("Complexity: Tree Size")
        plt.ylabel("F1 Score")
        plt.title("Best Pareto at Selected Generations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/Pop{pop}_Depth{depth}_Best_Pareto_Selected_Generations.png")
        plt.close()

        #9. Deduplicate Gen 50 again before final reporting
        gen_50_unique = []
        seen = set()
        for ind in best_pareto_fronts[49]:
            key = (str(ind), ind.fitness.values)
            if key not in seen:
                seen.add(key)
                gen_50_unique.append(ind)

        #10. Visualize the trees from the pareto front in the 50th gen and save to tree_visualizations
        for i, tree in enumerate(best_pareto_fronts[49]):
            save_best_tree_as_image(tree, f"{tree_dir}/Gen_50_Tree{i+1}")

        #11. Write each tree's function expression and its fitness metrics to a .txt file
        with open(os.path.join(base_dir, "Gen_50_Tree_Functions.txt"), "w") as f:
            for i, tree in enumerate(best_pareto_fronts[49]):
                f1_score_val, complexity = tree.fitness.values
                f.write(f"Individual {i + 1}:\n")
                f.write(f"Function: {str(tree)}\n")
                f.write(f"F1 Score: {f1_score_val:.4f}, Complexity (Tree Size): {int(complexity)}\n")
                f.write("-" * 40 + "\n")

        #12. Run the solutions of the best pareto fronts on the validation set
        print("\n=== Evaluating Pareto Front on Validation Set ===")

        #14. Send validation of trees run on 50th gen best pareto to a .txt file
        with open(os.path.join(base_dir, "Gen_50_Tree_Validation_Performance.txt"), "w") as f:
            for i, tree in enumerate(best_pareto_fronts[49]):
                print(f"\nEvaluating Tree {i+1}")
                
                # Compile tree
                tree_func = gp.compile(tree, pset=primitive_set)
                
                # Raw scores before thresholding
                raw_scores = [tree_func(**row) for row in X_test.to_dict(orient="records")]
                print(f"Raw scores (first 10): {raw_scores[:10]}")
                
                # Apply threshold
                predictions = [1 if score > tree.threshold else 0 for score in raw_scores]
                labels = y_test.tolist()
                
                # Log predictions vs labels
                print(f"Predictions (first 10): {predictions[:10]}")
                print(f"Labels (first 10):      {labels[:10]}")

                # Calculate F1 score
                f1 = f1_score(labels, predictions)
                
                # Write to file
                f.write(f"Individual {i+1}: \n")
                f.write(f"F1 Score (Validation): {f1:.4f}\n")
                f.write("-" * 40 + "\n")