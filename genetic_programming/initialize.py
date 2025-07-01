'''
File: initialize.py
The purpose of this file is to define the initialize_pop function, which generates an initial population of random primitive trees.
'''
from genetic_programming.toolbox import toolbox

def initialize_pop(pop_size):
    return [toolbox.individual() for i in range(pop_size)]
