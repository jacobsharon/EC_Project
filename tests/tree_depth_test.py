"""
This file tests for the optimal tree depth to be used in the main project by iterating through tree depths and calculating accuracy,
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import csv