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
from settings import POPULATION

#############################################################
# 1. GPModelWrapper: Callable wrapper around compiled trees #
#############################################################
class GPModelWrapper:
    """
    A wrapper class that allows DEAP-compiled symbolic trees to be used with SHAP.

    Parameters:
        func (callable): A compiled DEAP tree that takes named inputs and returns a score.

    Methods:
        predict(X_df): Applies the compiled tree to each row in a DataFrame and returns a NumPy array.
    """
    def __init__(self, func):
        self.func = func

    def predict(self, X_df):
        """
        Predicts outcomes using the compiled tree function for each row in a DataFrame.

        Parameters:
            self: Comiled tree wrapper
            X_df (pd.DataFrame): Feature input DataFrame.

        Returns:
            np.ndarray: Array of predicted outputs from the tree function.
        """
        return np.array([self.func(**row) for _, row in X_df.iterrows()])

#########################################################################
# 2. extract_tree_expression: Load function string from saved tree file #
#########################################################################
def extract_tree_expression(path):
    """
    Extracts the tree function string from a file that contains a 'Function:' line.

    Parameters:
        path (str): Path to the file containing the symbolic tree expression.

    Returns:
        str: The tree expression string.

    Raises:
        ValueError: If no line starting with 'Function:' is found.
    """
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Function:"):
                return line.split("Function:")[1].strip()
    raise ValueError("No 'Function:' line found in tree file.")

#######################################################
# 3. Load fold 1 temporarily to initialize data shape #
#######################################################
X_train, X_val, y_train, y_val = initialize_classification_model()[0]

########################################
# 4. Loop through each fold and model #
########################################

# 4.1 Outer loop: Iterate over all 5 folds
#     Each fold has its own trained model and corresponding SHAP output directory
for fold in range(1, 6):
    base_dir = f"results/final_results/fold_{fold}"                      # Root directory for current fold
    shap_dir = os.path.join(base_dir, "final_SHAP_analysis")             # Directory to save SHAP plots
    os.makedirs(shap_dir, exist_ok=True)                                 # Create SHAP directory if it doesn't exist

    # 4.2 Inner loop: Evaluate both Best and Elbow models from each fold
    for model_type in ["Best", "Elbow"]:
        tree_path = os.path.join(base_dir, f"{model_type}_Tree.txt")     # Path to saved symbolic tree
        if not os.path.exists(tree_path):
            print(f"Skipping {model_type} for Fold {fold}, file missing.")
            continue  # Skip if tree file is missing

        try:
            # 4.3 Load symbolic tree from saved text file
            tree_expr = extract_tree_expression(tree_path)

            # 4.4 Compile the tree into a Python function
            tree = gp.PrimitiveTree.from_string(tree_expr, primitive_set)
            compiled = gp.compile(tree, pset=primitive_set)

            # 4.5 Wrap compiled tree so SHAP can use it
            model = GPModelWrapper(compiled)

            # 4.6 Load the training and validation data for this fold
            X_train, X_val, y_train, y_val = initialize_classification_model()[fold - 1]    # fold -1 because fold was using 1-based index

            # 4.7 Generate SHAP values for the validation set
            explainer_val = shap.Explainer(model.predict, X_val)
            shap_vals_val = explainer_val(X_val)

            # 4.8 Generate and save SHAP summary plot
            shap.summary_plot(shap_vals_val, X_val, show=False)
            plt.title(f"SHAP Summary ({model_type}) Fold {fold}")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"{model_type}_Tree_SHAP.png"))
            plt.close()

        except Exception as e:
            print(f"Error in SHAP for {model_type} tree fold {fold}: {e}")
