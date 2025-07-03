'''
File: SHAP_analysis

The purpose of this file is to output a SHAP analysis on each of the solutions in the best pareto fronts from the 50th gen produced in main.py
To run shap execute "python -m explain.SHAP_analysis" in terminal. 
'''

# Standard Library
import os

# Third Party Libraries
import shap
import numpy as np
import matplotlib.pyplot as plt
from deap import gp

# Local Application Modules
from genetic_programming.primitives import primitive_set
from classification_model.classification_model import initialize_classification_model
from settings import MAX_DEPTH, POPULATION

# Wrapper for SHAP
class GPModelWrapper:
    def __init__(self, func):
        self.func = func

    def predict(self, X_df):
        return np.array([self.func(**row) for _, row in X_df.iterrows()])

# Extract tree strings from file
def extract_tree_expressions(path):
    tree_exprs = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Function:"):
                expr = line.split("Function:")[1].strip()
                tree_exprs.append(expr)
    return tree_exprs

# Load dataset
X_train, X_val, y_train, y_val = initialize_classification_model()[0]

# Iterate over each max depth setting
for depth in MAX_DEPTH:
    for pop_size in POPULATION:
        print(f"\n===== Running SHAP Analysis for pop {pop_size} max_depth {depth} =====")
        
        base_dir = f"results/pop_{pop_size}/max_depth_{depth}"
        tree_path = os.path.join(base_dir, "Gen_50_Tree_Functions.txt")
        shap_dir = os.path.join(base_dir, "SHAP_analysis")
        train_dir = os.path.join(shap_dir, "train")
        val_dir = os.path.join(shap_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Load trees
        try:
            tree_strs = extract_tree_expressions(tree_path)
        except FileNotFoundError:
            print(f"Missing: {tree_path}. Skipping depth {depth}")
            continue

        print(f"Parsed {len(tree_strs)} tree expressions")

        # Analyze each tree
        for i, expr in enumerate(tree_strs):
            try:
                print(f"Processing Tree {i+1}")
                tree = gp.PrimitiveTree.from_string(expr, primitive_set)
                compiled = gp.compile(tree, pset=primitive_set)
                model = GPModelWrapper(compiled)

                # SHAP on Training
                explainer_train = shap.Explainer(model.predict, X_train)
                shap_values_train = explainer_train(X_train)
                shap.summary_plot(shap_values_train, X_train, show=False)
                plt.savefig(f"{train_dir}/Tree{i+1}_shap_summary.png")
                plt.close()

                # SHAP on Validation
                explainer_val = shap.Explainer(model.predict, X_val)
                shap_values_val = explainer_val(X_val)
                shap.summary_plot(shap_values_val, X_val, show=False)
                plt.savefig(f"{val_dir}/Tree{i+1}_shap_summary.png")
                plt.close()

            except Exception as e:
                print(f"Error with Tree {i+1} at depth {depth}: {e}")