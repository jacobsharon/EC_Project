'''
File: primitives.py

The purpose of this file is to define functional and terminal sets for use in the genetic programming component of this project. 
Terminal Set - Features
Functional Set - Operations
Primitive Set  = Terminal Set + Functional Set
'''

# Standard Libraries
import operator
import math

# Third Party Libraries
from deap import gp

# Local Application Modules
from settings import feature_names

#1. Define primitive set with typed specification
primitive_set = gp.PrimitiveSetTyped("MAIN", [float] * len(feature_names), float)

#2. Rename default ARGs to class names stored in feature names
primitive_set.renameArguments(**{f'ARG{i}': name for i, name in enumerate(feature_names)})

#3. Safe function definitions
def if_then_else(condition: bool, out1: float, out2: float) -> float:
    return out1 if condition else out2

def protectedDiv(x: float, y: float) -> float:
    return x / y if y != 0 else 1

def protectedPow(x: float, y: float) -> float:
    try:
        return math.pow(abs(x), y)
    except:
        return 1

#4. Add arithmetic primitives
primitive_set.addPrimitive(operator.add, [float, float], float)
primitive_set.addPrimitive(operator.sub, [float, float], float)
primitive_set.addPrimitive(operator.mul, [float, float], float)
primitive_set.addPrimitive(protectedPow, [float, float], float)
primitive_set.addPrimitive(protectedDiv, [float, float], float)

#5. Unary numeric functions
primitive_set.addPrimitive(abs, [float], float)

#6. Comparison (return bool)
primitive_set.addPrimitive(operator.lt, [float, float], bool)
primitive_set.addPrimitive(operator.gt, [float, float], bool)
primitive_set.addPrimitive(operator.eq, [float, float], bool)
primitive_set.addPrimitive(operator.ne, [float, float], bool)

#7. Add boolean constants to support bool-returning functions
primitive_set.addTerminal(True, bool)
primitive_set.addTerminal(False, bool)

#8. Ternary logic
primitive_set.addPrimitive(if_then_else, [bool, float, float], float)