"""
This file contains global variables used within this project. 
"""
RANDOM_SEED = 103
POPULATION = [50,100]
GENERATIONS = 50
TOURNAMENT_SIZE = 2
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.5
MAX_DEPTH = [8,7,6]
K_FOLDS = 10

feature_names = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
    'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]