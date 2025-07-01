'''
File: initialize.py
The purpose of this file is to define the initialize_pop function,
which generates an initial population of random primitive trees.

Parameters: 
    pop_size (int): Number of individuals to generate.

Returns: 
    list: A list of deap.gp.PrimitiveTree individuals.
'''
from genetic_programming.toolbox import toolbox

def initialize_pop(pop_size):
    return [toolbox.individual() for _ in range(pop_size)]
