'''
File: primitives.py

The purpose of this file is to define functional and terminal sets for use in the genetic programming component of this project. 
Terminal Set - Features
Functional Set - Operations
Primitive Set  = Terminal Set + Functional Set
'''

from settings import feature_names
from deap import gp
import operator
from deap import base, creator
import math

#Define primitive set using DEAP function. "MAIN" is the name of the GP tree and the arity is the number of features, 24 + 1 classifier. 
primitive_set = gp.PrimitiveSet("MAIN" , len(feature_names))

#Iteratively rename the ARGs in the primitive set to match the feature names, this is the Terminal Set. 
primitive_set.renameArguments(**{f'ARG{i}': name for i, name in enumerate(feature_names)})

#safe function definitions for use in the primitive set
def if_then_else(condition, out1, out2):
    return out1 if condition else out2
def protectedDiv(x, y): return x / y if y != 0 else 1 #no div by 0
def protectedSqrt(x): return math.sqrt(abs(x)) #no sqrt neg numbers
def protectedLog(x): return math.log(abs(x)) if x != 0 else 0 #no log(0)
def protectedPow(x, y): #no domain issues
    try:
        return math.pow(abs(x), y)
    except:
        return 1

# Add functional set to primitive set
primitive_set.addPrimitive(operator.add, 2)
primitive_set.addPrimitive(operator.sub, 2)
primitive_set.addPrimitive(operator.mul, 2)
primitive_set.addPrimitive(protectedPow, 2)
primitive_set.addPrimitive(protectedDiv, 2)
primitive_set.addPrimitive(protectedSqrt, 1)
primitive_set.addPrimitive(protectedLog, 1)
primitive_set.addPrimitive(operator.neg, 1)
primitive_set.addPrimitive(abs, 1)
primitive_set.addPrimitive(operator.lt, 2)
primitive_set.addPrimitive(operator.gt, 2)
primitive_set.addPrimitive(operator.eq, 2)
primitive_set.addPrimitive(operator.ne, 2)
primitive_set.addPrimitive(if_then_else, 3)