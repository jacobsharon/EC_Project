'''
File: toolbox.py

The purpose of this file is to define the DEAP toolbox to be used in the evolutionary algorithm. 
'''

# Third Party Libraries
from deap import gp, base, tools, creator

# Local Application Modules
import genetic_programming.creator_setup
from genetic_programming.primitives import primitive_set

toolbox = base.Toolbox()

#Expression (tree) generator, generates an expression as a PrimitiveTree object
#named expr, generates genGrow and genFull 50/50 each time, calls the primitive set made in primitives.py
toolbox.register("expr", gp.genHalfAndHalf, pset = primitive_set, min_ = 2, max_ = 8)

#Individual generator
#creates an expression tree and then wraps it in an individual class that extends the PrimitiveTree class and includes fitness data
toolbox.register("individual" , tools.initIterate, creator.Individual, toolbox.expr)

#population generator
#iterates through the population generated and places them in a list to go to the genetic operators
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#Genetic Operators
#converts the PrimitiveTree expressions into a python interpretable function
toolbox.register("compile" , gp.compile, pset = primitive_set)

#Selection
toolbox.register("tournament_selection" , tools.selTournament, tournsize = 2)

#Crossover
toolbox.register("crossover" , gp.cxOnePoint)

#Mutations
#randomly selects a point on the tree and replaces the subtree with a randomly generated expression expr
toolbox.register("mutation", gp.mutUniform, expr=toolbox.expr, pset=primitive_set)

toolbox.register("clone", lambda ind: creator.Individual(ind))