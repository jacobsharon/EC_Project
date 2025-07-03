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

def initialize_classification_model():
    #1. read the cleaned ckd dataset.
    df = pd.read_csv("data/datasets/cleaned_ckd_dataset.csv")

    #2. Convert ckd notckd class to ints (had to debug because was feeding as str from cleaned dataset causing all f1=0)
    df["class"] = df["class"].astype(str).str.strip()
    y = df["class"].astype(int)

    #3. Drop the ckd notckd class from the list of features for X
    X = df.drop("class", axis=1).copy()

    #4. Identify numerical columns for scaling
    numerical_cols = []
    for col, type in feature_types.items():
        if type == "numerical":
            numerical_cols.append(col)

    #5. Call the scaler to scale numerical features [0,1]
    scaler = MinMaxScaler()
    
    #6. Fit transmorm X
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    #7. Split into 5 folds
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    print("Train CKD count:", sum(y_train))
    print("Val CKD count:", sum(y_val))
    
    #8. Store each fold in an array of folds
    return [(X_train, X_val, y_train, y_val)]