'''
File: creator_setup.py

Purpose: Define custom DEAP classes for symbolic genetic programming.
This initializes:
- FitnessMax: a fitness class where higher fitness is better (for maximization).
- Individual: a subclass of gp.PrimitiveTree with an associated FitnessMax.
'''

from deap import creator, base, gp

#######################################################################
# 1. Create multi-objective fitness and individual class for DEAP GP #
#######################################################################

# 1.1 Check if FitnessMulti is already defined; if not, create it
#     Maximizes F1 score (1.0) and minimizes tree complexity (-1.0)
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))

# 1.2 Check if Individual is already defined; if not, create it
#     Inherit from gp.PrimitiveTree and assign the custom fitness
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)