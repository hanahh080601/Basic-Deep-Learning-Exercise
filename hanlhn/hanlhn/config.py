import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
np.random.seed(42)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import hinge_loss, log_loss