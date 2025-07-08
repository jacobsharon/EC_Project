'''
File: initialize.py

The purpose of this file is to define the initialize_pop function, which generates an initial population of random primitive trees.
'''

from genetic_programming.toolbox import toolbox

#####################################################
# 1. initialize_pop: Generate random population set #
#####################################################
def initialize_pop(pop_size):
    """
    Initializes a population of individuals using the DEAP toolbox.

    Parameters:
        pop_size (int): The number of individuals to generate.

    Returns:
        list: A list of DEAP individuals representing the initial population.
    """
    # 1.1 Use the toolbox's individual generator to create a population of random trees
    return [toolbox.individual() for i in range(pop_size)]
