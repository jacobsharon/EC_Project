'''
File: genetic_programming/tree_validation.py
The purpose of this file is to verify trees are within type contraints defined in constraints.py
'''

from deap import gp
from genetic_programming.constraints import feature_types
from genetic_programming.primitives import primitive_set

def get_type(name):
    """
    Helper to get the feature type from the feature_types dict.
    Returns None if not found.
    """
    return feature_types.get(name)

def validate_tree(tree, feature_types):
    """
    Walks the tree in prefix order and validates type constraints.
    Returns True if the tree is type-consistent, False otherwise.
    """
    stack = []

    for node in reversed(tree):  # reversed for prefix traversal
        if isinstance(node, gp.Terminal):
            # Check if the terminal is a feature
            term_type = get_type(str(node.name)) if hasattr(node, 'name') else get_type(str(node))
            if term_type is None:
                return False  # unknown terminal node
            stack.append(term_type)
        else:
            # Check if the function expects arguments of consistent type
            arity = node.arity
            if len(stack) < arity:
                return False  # not enough arguments

            args = [stack.pop() for _ in range(arity)]
            # For simplicity, assume functions expect arguments of the same type
            if not all(arg == args[0] for arg in args):
                return False  # mismatched input types

            # Assume return type equals input type (common in GP)
            stack.append(args[0])  # push result type

    return len(stack) == 1  # only 1 consistent type should remain at the end