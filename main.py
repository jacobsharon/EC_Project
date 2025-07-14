'''
File: main.py

The purpose of this file is to serve as the main script for running the Evolutionary Computing Project (EC_Project)
SHAP analysis is executed separately (see SHAP_analysis.py) to avoid runtime overhead in main pipeline. 
To run shap execute "python -m explain.SHAP_analysis" in terminal. 
'''

# Standard library
import os

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, ConfusionMatrixDisplay
from deap import tools, gp
from scipy import stats

# Local application modules
from genetic_programming.toolbox import toolbox
from genetic_programming.initialize import initialize_pop
from genetic_programming.evaluate_population import evaluate_population
from genetic_programming.primitives import primitive_set
from classification_model.classification_model import initialize_classification_model
from results.visualize_tree import save_best_tree_as_image
from settings import GENERATIONS, POPULATION
from results.elbow_locator import find_elbow_point

###################################################################################
# 1. Initialize the classification model and array to store metrics for each fold #
###################################################################################
folds = initialize_classification_model()
fold_metrics = []

################################
# 2. Iterate through each fold #
################################
for fold_index, (X_train, X_test, y_train, y_test) in enumerate(folds):

    ##############################################
    # 3. Create result directories for this fold #
    ##############################################
    base_dir = f"results/final_results/fold_{fold_index + 1}"
    plot_dir = os.path.join(base_dir, "final_plots")
    tree_dir = os.path.join(base_dir, "final_tree_visualizations")
    shap_dir = os.path.join(base_dir, "final_SHAP_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(tree_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)

    #########################################################
    # 4. Initialize population and store best Pareto fronts #
    #########################################################
    population = initialize_pop(POPULATION)
    best_pareto_fronts = []

    ##################################
    # 5. Evaluate initial population #
    ##################################
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    if invalid_ind:
        evaluate_population(invalid_ind, X_train, y_train)

    ############################
    # 6. Run generational loop #
    ############################
    for gen in range(GENERATIONS):
        offspring = []

        # 6.1 Generate offspring until population size is met
        while len(offspring) < POPULATION:

            # 6.1.1 Select two parents via tournament selection
            parent1, parent2 = toolbox.tournament_selection(population, k=2, fit_attr="fitness")

            # 6.1.2 Clone the selected parents
            child1, child2 = map(toolbox.clone, (parent1, parent2))

            # 6.1.3 Delete fitness of clones
            del child1.fitness.values
            del child2.fitness.values

            # 6.1.4 Apply crossover to the cloned offspring
            toolbox.crossover(child1, child2)

            # 6.1.5 Apply mutation to each child
            child1, = toolbox.mutation(child1)
            child2, = toolbox.mutation(child2)

            # 6.1.6 Add children to the new offspring population
            offspring.extend([child1, child2])

        ##########################################
        # 7. Evaluate and select next generation #
        ##########################################

        # 7.1 Truncate offspring to maintain constant population size
        offspring = offspring[:POPULATION]

        # 7.2 Combine current population with offspring
        combined = population + offspring

        # 7.3 Identify individuals with invalid fitness
        invalid_ind = [ind for ind in combined if not ind.fitness.valid]

        # 7.4 Evaluate fitness of invalid individuals
        if invalid_ind:
            evaluate_population(invalid_ind, X_train, y_train)

        # 7.5 Select the next generation using NSGA-II (keep pop + offspring for elitism)
        population = tools.selNSGA2(population + offspring, k=POPULATION)

        ###################################################
        # 8. Extract unique individuals from Pareto front #
        ###################################################
        # 8.1 Get the first (non-dominated) Pareto front
        front = tools.sortNondominated(population, k=len(population), first_front_only=True)[0]

        # 8.2 Initialize list to store only unique individuals
        unique = []

        # 8.3 Initialize a set to track stringified individuals and their fitness
        seen = set()

        # 8.4 Iterate through individuals in the Pareto front
        for ind in front:
            key = (str(ind), ind.fitness.values)  # 8.5 Use string form and fitness values as a unique identifier

            # 8.6 Add only new individuals
            if key not in seen:
                seen.add(key)
                unique.append(ind)

        # 8.7 Replace front with list of unique individuals only
        front = unique

        # 8.8 Store this front in the list of best Pareto fronts across generations
        best_pareto_fronts.append(front)
        print(f"\n=== Population {POPULATION} | Generation {gen+1} completed ===")
        for child in population[:3]:
            f1, complexity = child.fitness.values
            terminals = [n for n in child if isinstance(n, gp.Terminal)]
            features = [t for t in terminals if hasattr(t, 'value') and isinstance(t.value, str)]
            print(f"Fitness: {f1:.4f} | Features: {int(complexity)}")

    ##############################
    # 9. Plot select generations #
    ##############################
    # 9.1 Define visual style to emphasize final generation (Gen 50) on the plot
    highlight_color = 'red'     # 50th Generation 
    highlight_size = 40
    normal_size = 40

    # 9.2 Iterate through 1, 10, 20, 30, 40, 50 pareto fronts and plot fitness vs complexity
    for i in [0, 9, 19, 29, 39, 49]:
        generation = best_pareto_fronts[i]
        f1_scores = [ind.fitness.values[0] for ind in generation]       # fitness.values is a tuple where [0] is f1 score
        complexities = [ind.fitness.values[1] for ind in generation]    # fitness.values is a tuple where [1] is complexity

        # 9.3 Plot all selected generations + the 50th gen
        if i == 49:
            plt.scatter(complexities, f1_scores,
                        label=f"Generation {i+1}",
                        color=highlight_color,
                        s=highlight_size,
                        edgecolor='black',
                        linewidths=1.2)
        else:
            plt.scatter(complexities, f1_scores,
                        label=f"Generation {i+1}",
                        s=normal_size,
                        alpha=0.6)

    # 9.4 Generate Plot of Pareto Fronts
    plt.xlabel("Complexity: Tree Size")
    plt.ylabel("F1 Score")
    plt.title(f"Fold {fold_index + 1}: Best Pareto at Selected Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Fold_{fold_index + 1}_Best_Pareto_at_Selected_Generations.png")
    plt.close()

    ################################################
    # 10. Determine elbow point from generation 50 #
    ################################################

    # 10.1 Create a list to store unique individuals from generation 50
    gen_50_unique = []

    # 10.2 Use a set to track seen individuals by their string representation and fitness
    #      This ensures that only distinct individuals are evaluated for the elbow point
    seen = set()

    # 10.3 Iterate over individuals in generation 50's Pareto front
    for ind in best_pareto_fronts[49]:
        key = (str(ind), ind.fitness.values)

        # 10.4 If the individual hasn't been seen, add it to the unique list
        #      This step prevents duplicated trees from skewing elbow detection
        if key not in seen:
            seen.add(key)
            gen_50_unique.append(ind)

    # 10.5 Sort the unique individuals by complexity (tree size), which is stored as fitness.values[1]
    #      Sorting to ensure a left-to-right trajectory for curve fitting
    gen_50_unique.sort(key=lambda ind: ind.fitness.values[1])

    # 10.6 Construct (x, y) coordinates for the elbow finder: (complexity, F1)
    pareto_points = [(ind.fitness.values[1], ind.fitness.values[0]) for ind in gen_50_unique]

    # 10.7 Call the elbow-finding function to identify the inflection point
    elbow_point = find_elbow_point(pareto_points)

    ########################################
    # 11. Locate elbow tree by coordinates #
    ########################################

    # Initialize placeholders in case the elbow point is not found.
    elbow_tree = None
    elbow_tree_index = -1

    # If an elbow point was successfully identified from the Pareto front:
    if elbow_point:
        # Search through unique individuals in generation 50 to find the one
        # whose fitness exactly (within tolerance) matches the elbow point coordinates.
        # These coordinates are (complexity, F1), so match [1] to F1 and [0] to complexity.
        # Compare (F1, Complexity) = fitness.values to elbow_point (x=Complexity, y=F1),
        # so we reverse access: fitness[0] to elbow_point[1], fitness[1] to elbow_point[0]
        for idx, ind in enumerate(gen_50_unique):
            if np.isclose(ind.fitness.values[0], elbow_point[1]) and np.isclose(ind.fitness.values[1], elbow_point[0]):
                elbow_tree = ind  # Store the matched individual
                try:
                    # Find the 1-based index of the elbow individual for output consistency (was getting 1 off individuals in results)
                    elbow_tree_index = gen_50_unique.index(elbow_tree) + 1
                except ValueError:
                    # If the individual isn't found, set index to -1
                    elbow_tree_index = -1
                break  # Stop once a match is found

    ###############################################
    # 12. Save tree visualizations and text files #
    ###############################################
    for i, tree in enumerate(gen_50_unique):
        save_best_tree_as_image(tree, f"{tree_dir}/Fold{fold_index + 1}_Individual{i + 1}")   # Use 1-based index for consistency

    with open(os.path.join(base_dir, "Gen_50_Tree_Functions.txt"), "w") as f:
        for i, tree in enumerate(gen_50_unique):
            f1_score_val, complexity = tree.fitness.values
            f.write(f"Individual {i + 1}:\n")
            f.write(f"Function: {str(tree)}\n")
            f.write(f"F1 Score: {f1_score_val:.4f}, Complexity (Tree Size): {int(complexity)}\n")
            f.write("-" * 40 + "\n")

    #########################################################################
    # 13. Evaluate all gen 50 individuals on test set and select best by F1 #
    #########################################################################

    # Initialize variables to track the best F1 score and corresponding model
    best_f1 = -1
    best_metrics = {}
    best_tree_index = -1

    # Loop through all unique individuals from generation 50
    for i, tree in enumerate(gen_50_unique):
        # Compile tree into an executable Python function
        tree_func = gp.compile(tree, pset=primitive_set)

        # Compute raw output scores by applying the model to each test instance
        raw_scores = [tree_func(**row) for row in X_test.to_dict(orient="records")]

        # Convert raw scores to binary predictions using the individual's learned threshold
        predictions = [1 if score > tree.threshold else 0 for score in raw_scores]

        # Extract true labels
        labels = y_test.tolist()

        # Compute confusion matrix components to derive specificity
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        specificity = tn / (tn + fp)

        # Calculate standard performance metrics for the individual
        metrics = {
            "F1": f1_score(labels, predictions),
            "Accuracy": accuracy_score(labels, predictions),
            "Sensitivity": recall_score(labels, predictions),
            "Specificity": specificity,
            "Precision": precision_score(labels, predictions),
            "AUC": roc_auc_score(labels, raw_scores),
        }

        # Check if this individual outperforms previous ones based on F1 score
        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]                          # Update best F1 score
            best_metrics = metrics.copy()                    # Store current best metrics
            best_metrics["Function"] = str(tree)             # Add function string to results
            best_tree = tree                                 # Store best tree object

            # Record the 1-based index of the best individual for reporting
            try:
                best_tree_index = gen_50_unique.index(best_tree) + 1    # use 1-based index for consistency 
            except ValueError:
                best_tree_index = -1  # In case an individual not found

    #################################################
    # 14. Save best model confusion matrix and file #
    #################################################

    # Recompile the best symbolic expression tree for prediction
    tree_func = gp.compile(best_tree, pset=primitive_set)

    # Apply the best model to test data to get raw output scores
    raw_scores = [tree_func(**row) for row in X_test.to_dict(orient="records")]

    # Convert scores to binary predictions using the model’s threshold
    predictions = [1 if score > best_tree.threshold else 0 for score in raw_scores]

    # Extract true labels from test set
    labels = y_test.tolist()

    # Compute confusion matrix components to calculate specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    specificity = tn / (tn + fp)

    # Add final metrics and model identifiers for this fold to the best_metrics dictionary
    best_metrics.update({
        "Fold": fold_index + 1,                          # 1-based index of current fold
        "Type": "Best",                                  # Indicates this is the best model
        "Function": str(best_tree),                      # String form of best tree
        "Individual": best_tree_index,                   # 1-based index in Gen 50 for consistency
        "F1": f1_score(labels, predictions),
        "Accuracy": accuracy_score(labels, predictions),
        "Sensitivity": recall_score(labels, predictions),
        "Specificity": specificity,
        "Precision": precision_score(labels, predictions),
        "AUC": roc_auc_score(labels, raw_scores),
    })

    # Plot and save a confusion matrix plot for the best model
    cm_best = confusion_matrix(labels, predictions)
    disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=["Not CKD", "CKD"])
    disp_best.plot(cmap="Blues", values_format='d')
    plt.title(f"Fold {fold_index + 1} Best Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Fold_{fold_index + 1}_Best_Confusion_Matrix.png")
    plt.close()

    # Save the best individual’s index and function string to a text file
    with open(os.path.join(base_dir, "Best_Tree.txt"), "w") as f:
        f.write(f"Individual {best_tree_index}:\n")
        f.write(f"Function: {str(best_tree)}\n\n")
        f.write("=== Evaluation Metrics ===\n")
        for k, v in best_metrics.items():
            if k not in {"Function", "Type"}:
                f.write(f"{k}: {v:.4f}\n")

    ###############################################
    # 15. Evaluate and store elbow model if found #
    ###############################################

    # If an elbow point was successfully identified in Step 11
    if elbow_tree:
        # Compile the elbow-point tree for evaluation
        tree_func = gp.compile(elbow_tree, pset=primitive_set)

        # Compute raw model outputs on test data
        raw_scores = [tree_func(**row) for row in X_test.to_dict(orient="records")]

        # Convert raw scores to binary predictions using model's threshold
        predictions = [1 if score > elbow_tree.threshold else 0 for score in raw_scores]

        # Extract true labels from test set
        labels = y_test.tolist()

        # Calculate specificity using confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        specificity = tn / (tn + fp)

        # Store all relevant evaluation metrics for the elbow-point model
        elbow_metrics = {
            "F1": f1_score(labels, predictions),
            "Accuracy": accuracy_score(labels, predictions),
            "Sensitivity": recall_score(labels, predictions),
            "Specificity": specificity,
            "Precision": precision_score(labels, predictions),
            "AUC": roc_auc_score(labels, raw_scores),
            "Fold": fold_index + 1,                   # Current fold index
            "Type": "Elbow",                          # Model type identifier
            "Function": str(elbow_tree),              # Stringified symbolic tree
            "Individual": elbow_tree_index            # 1-based index in generation
        }

        # Plot and save a confusion matrix for the elbow model
        cm_elbow = confusion_matrix(labels, predictions)
        disp_elbow = ConfusionMatrixDisplay(confusion_matrix=cm_elbow, display_labels=["Not CKD", "CKD"])
        disp_elbow.plot(cmap="Blues", values_format='d')
        plt.title(f"Fold {fold_index + 1} Elbow Point Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/Fold_{fold_index + 1}_Elbow_Confusion_Matrix.png")
        plt.close()

        # Save the elbow-point tree function to text file
        with open(os.path.join(base_dir, "Elbow_Tree.txt"), "w") as f:
            f.write(f"Individual {elbow_tree_index}:\n")
            f.write(f"Function: {str(elbow_tree)}\n\n")
            f.write("=== Evaluation Metrics ===\n")
            for k, v in elbow_metrics.items():
                if k not in {"Function", "Type"}:
                    f.write(f"{k}: {v:.4f}\n")

        # Store metrics for this fold to be used in overall summary
        fold_metrics.append(elbow_metrics)

    ##########################
    # 16. Store best metrics #
    ##########################
    fold_metrics.append(best_metrics)

####################################
# 17. Compute summary across folds #
####################################

# Print header to console
print("\n === Average Performance Across All Folds == ")

# Initialize a list to hold averaged metric results
average_metrics = []

# Identify all numeric keys in the metrics dictionary. Exclude string-based entries such as Function expression and Type
numeric_keys = [k for k in fold_metrics[0].keys() if k not in {"Function", "Type"}]

# Iterate through each numeric metric
for key in numeric_keys:
    # Extract the list of values for this metric across all folds
    values = [fold[key] for fold in fold_metrics]

    # Calculate mean and standard deviation for each metric
    mean = np.mean(values)
    std = np.std(values)

    # Store the results in a structured format for printing and file output
    average_metrics.append((key, mean, std))

# Save the averaged performance metrics to a text file for summary reporting
with open("results/final_results/5_fold_summary.txt", "w") as f:
    for name, mean, std in average_metrics:
        f.write(f"{name}: {mean:.4f} ± {std:.4f}\n")