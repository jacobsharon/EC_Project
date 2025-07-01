'''
File: classification_model.py
The purpose of this file is to generate the classification model using the cleaned_ckd_dataset.csv
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
from data.constraints import feature_types
from sklearn.preprocessing import MinMaxScaler
from settings import RANDOM_SEED

def initialize_classification_model():
    #1. read the cleaned ckd dataset.
    df = pd.read_csv("data/datasets/cleaned_ckd_dataset.csv")

    #2. Convert ckd notckd class to ints (had to debug because was feeding as str from cleaned dataset causing all f1=0)
    df["class"] = df["class"].astype(str).str.strip()
    y = df["class"].astype(int)

    #3. drop the ckd notckd class from the list of features for X
    X = df.drop("class", axis=1).copy()

    #4. identify numerical columns for scaling
    numerical_cols = []
    for col, type in feature_types.items():
        if type == "numerical":
            numerical_cols.append(col)

    #5. call the scaler to scale numerical features [0,1]
    scaler = MinMaxScaler()
    
    #6. Fit transmorm X
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    #7. split into 5 folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    #8. store each fold in an array of folds
    folds = []
    for train_index, test_index in skf.split(X, y):
        X_train_fold = X.iloc[train_index]
        X_test_fold = X.iloc[test_index]
        y_train_fold = y.iloc[train_index]
        y_test_fold = y.iloc[test_index]
        folds.append((X_train_fold, X_test_fold, y_train_fold, y_test_fold))

    return folds 