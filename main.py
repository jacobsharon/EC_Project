'''
File: main.py

The purpose of this file is to serve as the main script for running the Evolutionary Computing Project (EC_Project)
SHAP analysis is run separately in its own module due to time requirements. 
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

# 1. Initialize the classification model and array to store metrics for each fold
folds = initialize_classification_model()
fold_metrics = []

# 2. Iterate through each fold
for fold_index, (X_train, X_test, y_train, y_test) in enumerate(folds):

    # 3. Create result directories for this fold
    base_dir = f"results/final_results/fold_{fold_index + 1}"
    plot_dir = os.path.join(base_dir, "final_plots")
    tree_dir = os.path.join(base_dir, "final_tree_visualizations")
    shap_dir = os.path.join(base_dir, "final_SHAP_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(tree_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)

    # 4. Initialize population and store best Pareto fronts
    population = initialize_pop(POPULATION)
    best_pareto_fronts = []

    # 5. Evaluate initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    if invalid_ind:
        evaluate_population(invalid_ind, X_train, y_train)

    # 6. Run generational loop
    for gen in range(GENERATIONS):
        offspring = []
        while len(offspring) < POPULATION:
            parent1, parent2 = toolbox.tournament_selection(population, k=2, fit_attr="fitness")
            child1, child2 = map(toolbox.clone, (parent1, parent2))
            del child1.fitness.values
            del child2.fitness.values
            toolbox.crossover(child1, child2)
            child1, = toolbox.mutation(child1)
            child2, = toolbox.mutation(child2)
            offspring.extend([child1, child2])

        # 7. Evaluate and select next generation
        offspring = offspring[:POPULATION]
        combined = population + offspring
        invalid_ind = [ind for ind in combined if not ind.fitness.valid]
        if invalid_ind:
            evaluate_population(invalid_ind, X_train, y_train)

        population = tools.selNSGA2(population + offspring, k=POPULATION)

        # 8. Extract unique individuals from Pareto front
        front = tools.sortNondominated(population, k=len(population), first_front_only=True)[0]
        unique = []
        seen = set()
        for ind in front:
            key = (str(ind), ind.fitness.values)
            if key not in seen:
                seen.add(key)
                unique.append(ind)
        front = unique
        best_pareto_fronts.append(front)

        print(f"\n=== Population {POPULATION} | Generation {gen+1} completed ===")
        for child in population[:3]:
            f1, complexity = child.fitness.values
            terminals = [n for n in child if isinstance(n, gp.Terminal)]
            features = [t for t in terminals if hasattr(t, 'value') and isinstance(t.value, str)]
            print(f"Fitness: {f1:.4f} | Features: {int(complexity)}")

    # 9. Plot select generations
    highlight_color = 'red'
    highlight_size = 40
    normal_color = 'blue'
    normal_size = 40

    for i in [0, 9, 19, 29, 39, 49]:
        generation = best_pareto_fronts[i]
        f1_scores = [ind.fitness.values[0] for ind in generation]
        complexities = [ind.fitness.values[1] for ind in generation]

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

    plt.xlabel("Complexity: Tree Size")
    plt.ylabel("F1 Score")
    plt.title(f"Fold {fold_index + 1}: Best Pareto at Selected Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Fold_{fold_index + 1}_Best_Pareto_at_Selected_Generations.png")
    plt.close()

    # 10. Determine elbow point from generation 50
    gen_50_unique = []
    seen = set()
    for ind in best_pareto_fronts[49]:
        key = (str(ind), ind.fitness.values)
        if key not in seen:
            seen.add(key)
            gen_50_unique.append(ind)

    gen_50_unique.sort(key=lambda ind: ind.fitness.values[1])
    pareto_points = [(ind.fitness.values[1], ind.fitness.values[0]) for ind in gen_50_unique]
    elbow_point = find_elbow_point(pareto_points)

    # 11. Locate elbow tree by coordinates
    elbow_tree = None
    elbow_tree_index = -1
    if elbow_point:
        for idx, ind in enumerate(gen_50_unique):
            if np.isclose(ind.fitness.values[0], elbow_point[1]) and np.isclose(ind.fitness.values[1], elbow_point[0]):
                elbow_tree = ind
                try:
                    elbow_tree_index = gen_50_unique.index(elbow_tree) + 1
                except ValueError:
                    elbow_tree_index = -1
                break

    # 12. Save tree visualizations and text files
    for i, tree in enumerate(best_pareto_fronts[49]):
        save_best_tree_as_image(tree, f"{tree_dir}/Gen_50_Tree{i+1}")

    with open(os.path.join(base_dir, "Gen_50_Tree_Functions.txt"), "w") as f:
        for i, tree in enumerate(gen_50_unique):
            f1_score_val, complexity = tree.fitness.values
            f.write(f"Individual {i + 1}:\n")
            f.write(f"Function: {str(tree)}\n")
            f.write(f"F1 Score: {f1_score_val:.4f}, Complexity (Tree Size): {int(complexity)}\n")
            f.write("-" * 40 + "\n")

    # 13. Evaluate all gen 50 individuals on test set and select best by F1
    best_f1 = -1
    best_metrics = {}
    best_tree_index = -1

    for i, tree in enumerate(gen_50_unique):
        tree_func = gp.compile(tree, pset=primitive_set)
        raw_scores = [tree_func(**row) for row in X_test.to_dict(orient="records")]
        predictions = [1 if score > tree.threshold else 0 for score in raw_scores]
        labels = y_test.tolist()
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        specificity = tn / (tn + fp)

        metrics = {
            "F1": f1_score(labels, predictions),
            "Accuracy": accuracy_score(labels, predictions),
            "Sensitivity": recall_score(labels, predictions),
            "Specificity": specificity,
            "Precision": precision_score(labels, predictions),
            "AUC": roc_auc_score(labels, raw_scores),
        }

        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            best_metrics = metrics.copy()
            best_metrics["Function"] = str(tree)
            best_tree = tree
            try:
                best_tree_index = gen_50_unique.index(best_tree) + 1
            except ValueError:
                best_tree_index = -1

    # 14. Save best model confusion matrix and file
    tree_func = gp.compile(best_tree, pset=primitive_set)
    raw_scores = [tree_func(**row) for row in X_test.to_dict(orient="records")]
    predictions = [1 if score > best_tree.threshold else 0 for score in raw_scores]
    labels = y_test.tolist()
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    specificity = tn / (tn + fp)

    best_metrics.update({
        "Fold": fold_index + 1,
        "Type": "Best",
        "Function": str(best_tree),
        "Individual": best_tree_index,
        "F1": f1_score(labels, predictions),
        "Accuracy": accuracy_score(labels, predictions),
        "Sensitivity": recall_score(labels, predictions),
        "Specificity": specificity,
        "Precision": precision_score(labels, predictions),
        "AUC": roc_auc_score(labels, raw_scores),
    })

    cm_best = confusion_matrix(labels, predictions)
    disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=["Not CKD", "CKD"])
    disp_best.plot(cmap="Blues", values_format='d')
    plt.title(f"Fold {fold_index + 1} Best Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Fold_{fold_index + 1}_Best_Confusion_Matrix.png")
    plt.close()

    with open(os.path.join(base_dir, "Best_Tree.txt"), "w") as f:
        f.write(f"Individual {best_tree_index}:\n")
        f.write(f"Function: {str(best_tree)}\n")

    # 15. Evaluate and store elbow model if found
    if elbow_tree:
        tree_func = gp.compile(elbow_tree, pset=primitive_set)
        raw_scores = [tree_func(**row) for row in X_test.to_dict(orient="records")]
        predictions = [1 if score > elbow_tree.threshold else 0 for score in raw_scores]
        labels = y_test.tolist()
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        specificity = tn / (tn + fp)

        elbow_metrics = {
            "F1": f1_score(labels, predictions),
            "Accuracy": accuracy_score(labels, predictions),
            "Sensitivity": recall_score(labels, predictions),
            "Specificity": specificity,
            "Precision": precision_score(labels, predictions),
            "AUC": roc_auc_score(labels, raw_scores),
            "Fold": fold_index + 1,
            "Type": "Elbow",
            "Function": str(elbow_tree),
            "Individual": elbow_tree_index
        }

        cm_elbow = confusion_matrix(labels, predictions)
        disp_elbow = ConfusionMatrixDisplay(confusion_matrix=cm_elbow, display_labels=["Not CKD", "CKD"])
        disp_elbow.plot(cmap="Blues", values_format='d')
        plt.title(f"Fold {fold_index + 1} Elbow Point Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/Fold_{fold_index + 1}_Elbow_Confusion_Matrix.png")
        plt.close()

        with open(os.path.join(base_dir, "Elbow_Tree.txt"), "w") as f:
            f.write(f"Individual {elbow_tree_index}:\n")
            f.write(f"Function: {str(elbow_tree)}\n")

        fold_metrics.append(elbow_metrics)

    # 16. Store best metrics
    fold_metrics.append(best_metrics)

# 17. Compute summary across folds
print("\n === Average Performance Across All Folds == ")
average_metrics = []

numeric_keys = [k for k in fold_metrics[0].keys() if k not in {"Function", "Type"}]
for key in numeric_keys:
    values = [fold[key] for fold in fold_metrics]
    mean = np.mean(values)
    std = np.std(values)
    average_metrics.append((key, mean, std))

with open("results/final_results/5_fold_summary.txt", "w") as f:
    for name, mean, std in average_metrics:
        print(f"{name}: {mean:.4f} ± {std:.4f}")
        f.write(f"{name}: {mean:.4f} ± {std:.4f}\n")