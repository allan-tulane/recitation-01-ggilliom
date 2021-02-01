import sklearn
import numpy as np
import pandas as pd

# A regression model that can compute rms and stuff.
from sklearn.linear_model import LinearRegression

# Load data from table.
table_data = pd.DataFrame({
	"n": [10, 100, 1000, 10000, 100000, 1000000, 10000000],
	"linear": [0.005, 0.011, 0.105, 1.211, 11.100, 251.453, 2101.483],
	"binary": [0.004, 0.014, 0.008, 0.018, 0.018, 0.033, 0.060]
})

# Compute log_2(n).
table_data["log2_n"] = table_data.n.map(np.log2)

# Perform linear regression for linear search.
X = table_data.n.values[:,None]
y = table_data.linear.values

# Regress and report.
lr_linear_search = LinearRegression()
lr_linear_search.fit(X, y)
print("Linear search R^2 Value: {}".format(
	lr_linear_search.score(X, y)))

# Perform linear regression for binary search.
X = table_data.log2_n.values[:,None]
y = table_data.binary.values

# Regress and report
lr_binary_search = LinearRegression()
lr_binary_search.fit(X, y)
print("Binary search R^2 Value: {}".format(
	lr_binary_search.score(X, y)))





