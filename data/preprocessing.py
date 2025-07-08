'''
File: preprocessing.py
The purpose of this file is to read the ckd_dataset.csv and perform data cleaning. 
Missing values handled by replacing with the average of the dataset or as treating the patient without a diagnosis for the given attribute.

Data Cleaning and Imputation of Missing Values
----------------------------------------------
Age (Numerical): Replace missing values with the mean of the dataset feature
Blood Pressure (bp) (Numerical): Replace missing values with the mean of the dataset feature
Specific Gravity (sg) (Nominal): Replace missing values with the mode of the dataset feature
Sugar (su) (Nominal): Replace missing values with the mode of the dataset feature
Albumin (al) (Nominal): Replace missing values with the mode of the dataset feature
Red Blood Cells (rbc) (Categorical): Replace missing values with 'normal'
Pus Cell (pc) (Categorical): Replace missing values with 'normal'
Pus Cell Clumps (pcc) (Categorical): Replace missing values with 'notpresent'
Bacteria (ba) (Categorical): Replace missing values with 'notpresent'
Blood Glucose Random (bgr) (Numerical): Replace missing values with the mean of the dataset feature
Blood Urea (bu) (Numerical): Replace missing values with the mean of the dataset feature
Serum Creatinine (sc) (Numerical): Replace missing values with the mean of the dataset feature
Sodium (sod) (Numerical): Replace missing values with the mean of the dataset feature
Potassium (pot) (Numerical): Replace missing values with the mean of the dataset feature
Hemoglobin (hemo) (Numerical): Replace missing values with the mean of the dataset feature
Packed Cell Volume (pcv) (Numerical): Replace missing values with the mean of the dataset feature
White Blood Cell Count (wc) (Numerical): Replace missing values with the mean of the dataset feature
Red Blood Cell Count (rc) (Numerical): Replace missing values with the mean of the dataset feature
Hypertension (htn) (Categorical): Replace missing values with 'no'
Diabetes Mellitus (dm) (Categorical): Replace missing values with 'no'
Coronary Artery Disease (cad) (Categorical): Replace missing values with 'no' 
Appetite (appet) (Categorical): Replace missing values with 'good'
Pedal Edema (pe) (Categorical): Replace missing values with 'no'
Anemia (ane) (Categorical): Replace missing values with 'no'
CKD (class) (Categorical): No missing values
'''

# Standard Libraries
import csv

# Third Party Libraries
import pandas as pd
import numpy as np

# Local Application Modules
from constraints import numerical_value_constraints_dict

file_path = "/Users/jacobsharon/Documents/Masters Degree/Summer 2025/CSC742/Project/EC_Project/data/datasets/raw_ckd_dataset.csv"

##########################################################
# 1. Read and preprocess CKD dataset for downstream use #
##########################################################

# Open the raw CKD dataset CSV file
with open(file_path, "r") as CKD:

    # 1.1 Initialize arrays for storing data and headers
    data_rows = []
    column_names = []

    # 1.2 Define numeric columns used for mean imputation
    numeric_columns = [
        'age', 'bp', 'bgr', 'bu', 'sc', 
        'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

    # 1.3 Define rounding precision for formatted averages
    format_map = {
        'age': 1,
        'bp': 0,
        'sg': 3,
        'al': 0,
        'su': 0,
        'bgr': 0,
        'bu': 1,
        'sc': 1,
        'sod': 1,
        'pot': 1,
        'hemo': 1,
        'pcv': 0,
        'wc': 0,
        'rc': 2,
    }

    # 1.4 Read file contents and strip whitespace
    lines = CKD.readlines()
    for i, line in enumerate(lines):
        cleaned_line = line.strip()
        if cleaned_line:
            if i == 0:
                column_names = cleaned_line.split(',')
            else:
                data_rows.append(cleaned_line.split(','))

    # 1.5 Convert parsed data into a DataFrame
    df = pd.DataFrame(data_rows, columns=column_names)
    df.replace(["?", ""], np.nan, inplace=True)

    # 1.6 Compute column-wise means for numeric attributes
    column_means = df[numeric_columns].apply(pd.to_numeric, errors='coerce').mean()

    # 1.7 Format means based on precision map
    formatted_means = {}
    for column, mean in column_means.items():
        if column in format_map:
            rounded_value = round(mean, format_map[column])
            formatted_means[column] = rounded_value
        else:
            formatted_means[column] = mean

    # 1.8 Write formatted means to output CSV
    with open('/Users/jacobsharon/Documents/Masters Degree/Summer 2025/CSC742/Project/EC_Project/data/default_attribute_values/numeric_attribute_averages.csv', 'w', newline="") as avg:
        avg_writer = csv.writer(avg)
        avg_writer.writerow(["attribute", "formatted_mean"])
        avg_writer.writerows(formatted_means.items())

    # 1.9 Read mean values and apply imputation to numeric columns
    means_df = pd.read_csv("data/default_attribute_values/numeric_attribute_averages.csv")
    means_dict = dict(zip(means_df["attribute"], means_df["formatted_mean"]))

    for col in means_dict:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(value=means_dict, inplace=True)

    # 1.10 Replace missing categorical values using predefined defaults
    default_cat_df = pd.read_csv("data/default_attribute_values/categorical_attribute_defaults.csv")
    default_cat_dict = dict(zip(default_cat_df["attribute"], default_cat_df["default_value"]))
    for col, default_value in default_cat_dict.items():
        df[col] = df[col].fillna(default_value)

    # 1.11 Replace missing nominal values using most frequent class (mode)
    default_nom_df = pd.read_csv("data/default_attribute_values/nominal_attribute_defaults.csv")
    default_nom_dict = dict(zip(default_nom_df["attribute"], default_nom_df["default_value"]))
    for col, default_value in default_nom_dict.items():
        df[col] = df[col].fillna(default_value)

    # 1.12 Encode categorical and nominal attributes to integer codes
    encode_columns = ['al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']
    for col in encode_columns:
        df[col] = pd.Categorical(df[col]).codes

    # 1.13 Apply constraints for numerical features defined in constraints.py
    for col, (min_value, max_values) in numerical_value_constraints_dict.items():
        if col in df[col]:
            df = df[df[col].between(min_value, max_values)]

    # 1.14 Export the final cleaned dataset to CSV for use in modeling
    df.to_csv("data/datasets/cleaned_ckd_dataset.csv", index=False)
