'''
File: classification_model.py
The purpose of this file is to generate the classification model using the cleaned_ckd_dataset.csv
'''

from sklearn.model_selection import StratifiedKFold as SKF
import pandas as pd
from settings import RANDOM_SEED
from genetic_programming.constraints import feature_types
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/datasets/cleaned_ckd_dataset.csv")

#X is the array of all independent variables, so drop the final classifier
#y is the array of dependent variables, so the final classifier (ckd, notckd)
X = df.drop("class" , axis=1)
y = df["class"]

#identify numerical columns for scaling
numerical_cols = []
for col, type in feature_types.items():
    if type == "numerical":
        numerical_cols.append(col)

#call the scaler
scaler = MinMaxScaler()

X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

stratified_k_folds = SKF(n_splits=10, shuffle=True, random_state=RANDOM_SEED)