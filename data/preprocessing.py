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

#1. Open the file
with open(file_path, "r") as CKD:
    #2. create arrays of data and attributes
    data_rows = []
    column_names = []
    
    #specify numeric attributes to calculate averages
    numeric_columns = [
    'age', 'bp', 'bgr', 'bu', 'sc', 
    'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

    #formatting guide for numeric attributes
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
    'rc': 2,}

    #3. add each line of the csv to an array
    lines = CKD.readlines()

    #4. iterate through each line and strip white space
    for i, line in enumerate(lines):
        cleaned_line = line.strip()
        if cleaned_line:
            if i == 0: # if it is the first element of a column it is the attribute name
                column_names = cleaned_line.split(',') 
            else:
                data_rows.append(cleaned_line.split(','))
    
    #5. convert the rows and columns to a dataframe
    df = pd.DataFrame(data_rows, columns=column_names)
    df.replace(["?", ""], np.nan, inplace=True)  # Convert '?' and empty string to actual NaN *before* processing

    #6. calculate the numeric column means
    column_means = df[numeric_columns].apply(pd.to_numeric, errors='coerce').mean()

    #7. format means to proper precision using the formatting guide
    formatted_means = {}
    for column, mean in column_means.items(): # Use .items() to iterate over Series
        if column in format_map:
            rounded_value = round(mean, format_map[column])
            formatted_means[column] = rounded_value
        else:
            formatted_means[column] = mean # Keep original if no formatting specified

    #8. output formatted means to data file
    with open('/Users/jacobsharon/Documents/Masters Degree/Summer 2025/CSC742/Project/EC_Project/data/default_attribute_values/numeric_attribute_averages.csv' , 'w', newline="") as avg:
        avg_writer = csv.writer(avg)
        avg_writer.writerow(["attribute", "formatted_mean"])
        avg_writer.writerows(formatted_means.items())
    
    #9. Replace the missing numeric values with the average
    means_df = pd.read_csv("data/default_attribute_values/numeric_attribute_averages.csv")
    means_dict = dict(zip(means_df["attribute"], means_df["formatted_mean"]))

    #10. Ensure numeric conversion for target columns
    for col in means_dict:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #11. Apply fillna once with all averages
    df.fillna(value=means_dict, inplace=True)

    #12. Replace the missing categorical values with their respective normal values
    default_cat_df = pd.read_csv("data/default_attribute_values/categorical_attribute_defaults.csv")
    default_cat_dict = dict(zip(default_cat_df["attribute"], default_cat_df["default_value"]))
    for col, default_value in default_cat_dict.items():
        df[col] = df[col].fillna(default_value)

    #13. Replace the missing nominal values with the mode
    default_nom_df = pd.read_csv("data/default_attribute_values/nominal_attribute_defaults.csv")
    default_nom_dict = dict(zip(default_nom_df["attribute"] , default_nom_df["default_value"]))
    for col, default_value in default_nom_dict.items():
        df[col] = df[col].fillna(default_value)

    # 14. Encode nominal features as 0 or 1
    encode_columns = ['al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']
    for col in encode_columns:
        df[col] = pd.Categorical(df[col]).codes

    #15. Enforce numerical feature constrains
    for col, (min_value, max_values) in numerical_value_constraints_dict.items():
        if col in df[col]:
            df = df[df[col].between(min_value, max_values)]

    #15. Output final cleaned dataset
    df.to_csv("data/datasets/cleaned_ckd_dataset.csv", index=False)