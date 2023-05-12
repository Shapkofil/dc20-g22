import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import plotly.graph_objects as go
from scipy.stats import pearsonr
import numpy as np

# Paths and reading paths
filepath_stopandsearch = "C:/Metropolitan-stopandsearch.parquet"
filepath_outcomes = "C:/Metropolitan-outcomes.parquet"
filepath_barnet_only = "C:/BarnetOnly.parquet"
filepath_street = "C:/Metropolitan-street.parquet"
filepath_barnetwith_out = "C:/barnetWith_Out.parquet"

#Switch all to pandas
stopandsearch = pq.read_pandas(filepath_stopandsearch).to_pandas()
outcomes = pq.read_pandas(filepath_outcomes).to_pandas()
barnet_only = pq.read_pandas(filepath_barnet_only).to_pandas()
street = pq.read_pandas(filepath_street).to_pandas()
barnet_with_out = pq.read_pandas(filepath_barnetwith_out).to_pandas()

# Subsetting the data
barnet = street[street["LSOA name"].str.contains("Barnet")==True]

# Correlation
id_and_crimes = barnet[["Crime ID", "Crime type"]]

# Pivot the data to create a binary matrix of crimes
crime_matrix = id_and_crimes.pivot_table(index='Crime ID', columns='Crime type', aggfunc=len, fill_value=0)

# Calculate the correlation matrix
correlation_matrix = crime_matrix.corr()
np.fill_diagonal(correlation_matrix.values, 0) # Set the diagonal to 0 to make the plot look better
correlation_matrix

#Plot
import seaborn as sns
import matplotlib.pyplot as plt
# Create a heatmap of the correlation matrix
plt.figure(figsize=(16, 14))

cmap = sns.diverging_palette(10, 240)
sns.heatmap(correlation_matrix, annot=True, center=0,cmap='bwr', linewidths=0.5, vmin=-1, vmax=1)
plt.title("Crime Type Correlation Matrix")
plt.xlabel("Crime Type")
plt.ylabel("Crime Type")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
