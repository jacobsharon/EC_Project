'''
File: constraints.py
The purpose of this file is to define:
1. Upper and lower bounds of numerical features based on diagnostic feasibility.
2. Allowed nominal and binary values for categorical features.
3. Feature type classification (numerical, nominal, binary) to support type-aware operations in genetic programming.

-------------------
Numerical Features 
-------------------
Age (years): Reference ranges for pediatric patients are not available and calculation of eGFR is based on adult studies. Patients <18 are excluded from this model.
     eGFR decreases gradually with age beyond age 40. Patients >65 in age have markedly decreased eGFR that are not indicative of CKD. Patients >65 are excluded from this model.

Blood Pressure (bp in mm/Hg): Minimum and Maximum of the dataset. Any BP > 0 (alive).

Specific Gravity (sg, unitless): Minimum and Maximum of the dataset. All values in the dataset are within normal to elevated physiological range (1.000, 1.050].

Blood Glucose Random (bgr in mgs/dl): Minimum and Maximum of the dataset. All values are within possible physiological range. Diabetes Mellitus is a known contributor to kidney damage.

Blood Urea (bu in mgs/dl): Minimum and maximum of the dataset. All values are within possible physiological ranges.

Serum Creatinine (sc in mgs/dl): Minimum and maximum of the dataset. All values are within possible physiological ranges.

Sodium (sod in mEq/L): Limiting to the lower and upper ends of critical value reference ranges.

Potassium (pot in mEq/L): Limiting to the lower and upper ends of critical value reference ranges.

Hemoglobin (hemo in gms): Limiting to the lower and upper ends of critical value reference ranges.

Packed Cell Volume (pcv in %): AKA Hematocrit - Limiting to the lower and upper ends of critical value reference ranges.

White Blood Cell Count (wc in cells/cumm): Minimum and maximum of the dataset. All values are within possible physiological ranges.

Red Blood Cell Count (rc in millions/cmm): Minimum and maximum of the dataset. All values are within possible physiological ranges.

----------------
Nominal Features
----------------

Enforced as collected in the dataset

'''

numerical_value_constraints_dict = {
    'age' : (18,65),
    'bp': (50,200),
    'sg': (1.005, 1.050),
    'bgr': (20,1000),
    'bu': (1.500, 500.00),
    'sc': (0.100, 100.00),
    'sod': (100, 170),
    'pot': (2.00,10.00),
    'hemo': (3.00,22.00),
    'pcv': (30.00, 55.00),
    'wc': (900, 250000),
    'rc': (2.00, 10.00)
}

nominal_values_constraints_dict = {
    'al': [0, 1, 2, 3, 4, 5],
    'su': [0, 1, 2, 3, 4, 5],
    'rbc': ['normal', 'abnormal'],
    'pc': ['normal', 'abnormal'],
    'pcc': ['present', 'notpresent'],
    'ba': ['present', 'notpresent'],
    'htn': ['yes', 'no'],
    'dm': ['yes', 'no'],
    'cad': ['yes', 'no'],
    'appet': ['good', 'poor'],
    'pe': ['yes', 'no'],
    'ane': ['yes', 'no'],
    'class': ['ckd', 'notckd']
}

feature_types = {
    'age': 'numerical',
    'bp': 'numerical',
    'sg': 'numerical',
    'al': 'nominal',
    'su': 'nominal',
    'rbc': 'binary',
    'pc': 'binary',
    'pcc': 'binary',
    'ba': 'binary',
    'bgr': 'numerical',
    'bu': 'numerical',
    'sc': 'numerical',
    'sod': 'numerical',
    'pot': 'numerical',
    'hemo': 'numerical',
    'pcv': 'numerical',
    'wc': 'numerical',
    'rc': 'numerical',
    'htn': 'binary',
    'dm': 'binary',
    'cad': 'binary',
    'appet': 'binary',
    'pe': 'binary',
    'ane': 'binary',
    'class': 'binary'
}