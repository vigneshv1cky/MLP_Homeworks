import pandas as pd
import numpy as np
import copy
import re
import matplotlib.pyplot as plt

import pickle as pkl

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from IPython import display


##################
# Default parameters
FIGURESIZE = (10, 6)
FONTSIZE = 10

plt.rcParams["figure.figsize"] = FIGURESIZE
plt.rcParams["font.size"] = FONTSIZE + 2

plt.rcParams["xtick.labelsize"] = FONTSIZE
plt.rcParams["ytick.labelsize"] = FONTSIZE
