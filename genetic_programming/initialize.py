'''
File: initialize.py
The purpose of this file is to define the initialize_pop function,
which generates an initial population of random primitive trees.

Parameters: 
    pop_size (int): Number of individuals to generate.

Returns: 
    list: A list of deap.gp.PrimitiveTree individuals.
'''
from genetic_programming.primitives import primitive_set
from deap import gp

#Define initialize_pop: takes parameter pop_size to get intial population size
def initialize_pop(pop_size):
    #create empty arrays to store the initial population and weights
    initial_population = []

    #randomly select one of the order sizes from the array and create new arrays with the order size indicating the max bins
    for i in range(pop_size):
        expr = gp.genFull(primitive_set, min_=3, max_=8)
        tree = gp.PrimitiveTree(expr)
        initial_population.append(tree)
    return initial_population