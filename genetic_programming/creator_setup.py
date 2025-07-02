'''
File: creator_setup.py
Purpose: Define custom DEAP classes for symbolic genetic programming.
This initializes:
- FitnessMax: a fitness class where higher fitness is better (for maximization).
- Individual: a subclass of gp.PrimitiveTree with an associated FitnessMax.
'''

from deap import creator, base, gp

#1. Check if FitnessMax is already defined; if not, create it as a maximization fitness and minimization features class
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))

#2. Check if Individual is already defined; if not, create it as a PrimitiveTree with FitnessMax as its fitness attribute
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)