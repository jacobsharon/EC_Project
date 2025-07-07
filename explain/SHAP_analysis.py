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

# Extract tree string from Function: line
def extract_tree_expression(path):
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Function:"):
                return line.split("Function:")[1].strip()
    raise ValueError("No 'Function:' line found in tree file.")

# Load fold 1 data temporarily to initialize structure
X_train, X_val, y_train, y_val = initialize_classification_model()[0]

# Loop through each fold
for fold in range(1, 6):
    base_dir = f"results/final_results/fold_{fold}"
    shap_dir = os.path.join(base_dir, "final_SHAP_analysis")
    os.makedirs(shap_dir, exist_ok=True)

    for model_type in ["Best", "Elbow"]:
        tree_path = os.path.join(base_dir, f"{model_type}_Tree.txt")
        if not os.path.exists(tree_path):
            print(f"Skipping {model_type} for Fold {fold}, file missing.")
            continue

        try:
            # Only read actual tree expression
            tree_expr = extract_tree_expression(tree_path)

            # Compile and wrap tree
            tree = gp.PrimitiveTree.from_string(tree_expr, primitive_set)
            compiled = gp.compile(tree, pset=primitive_set)
            model = GPModelWrapper(compiled)

            # Load data for current fold
            X_train, X_val, y_train, y_val = initialize_classification_model()[fold - 1]

            # SHAP on validation set
            explainer_val = shap.Explainer(model.predict, X_val)
            shap_vals_val = explainer_val(X_val)
            shap.summary_plot(shap_vals_val, X_val, show=False)
            plt.title(f"SHAP Summary ({model_type}) Fold {fold}")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"{model_type}_Tree_SHAP.png"))
            plt.close()

        except Exception as e:
            print(f"Error in SHAP for {model_type} tree fold {fold}: {e}")