'''
File: classification_model.py

The purpose of this file is to generate the classification model using the cleaned_ckd_dataset.csv
'''

# Third Party Libraries
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Local Application Modules
from data.constraints import feature_types
from settings import RANDOM_SEED

#############################################################
# 1. initialize_classification_model: Preprocess and Split #
#############################################################
def initialize_classification_model():
    """
    Loads the cleaned CKD dataset, encodes labels, scales numerical features,
    and performs stratified 5-fold cross-validation.

    Returns:
        list of tuples: Each tuple contains (X_train, X_val, y_train, y_val) 
                        for one of the 5 cross-validation folds.
    """
    
    # 1. Read the cleaned CKD dataset
    df = pd.read_csv("data/datasets/cleaned_ckd_dataset.csv")

    # 2. Convert 'class' values (ckd, notckd) to integers
    #    This fixes a bug where string values caused F1 score to be 0
    df["class"] = df["class"].astype(str).str.strip()
    y = df["class"].astype(int)

    # 3. Drop the target variable from feature set
    X = df.drop("class", axis=1).copy()

    # 4. Identify numerical columns for scaling
    numerical_cols = []
    for col, type in feature_types.items():
        if type == "numerical":
            numerical_cols.append(col)

    # 5. Initialize a MinMaxScaler to scale features between 0 and 1
    scaler = MinMaxScaler()

    # 6. Fit and transform only numerical columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # 7. Create a stratified 5-fold splitter to maintain class distribution
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # 8. Initialize list to hold all folds
    folds = []

    # 9. Iterate through splits and store training and validation sets
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]     # Access rows via integer index
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        folds.append((X_train, X_val, y_train, y_val))              # Append the current fold as a tuple

        print("Train CKD count:", sum(y_train))                     # Print class balance for debugging
        print("Val CKD count:", sum(y_val))

    # 10. Return list of (X_train, X_val, y_train, y_val) for each fold
    return folds