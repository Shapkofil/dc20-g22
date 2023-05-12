import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import plotly.graph_objects as go

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
barnetBur_df = barnet[barnet['Crime type'].str.contains('Burglary')==True]
barnet2020 = barnetBur_df[barnetBur_df["Month"].str.contains("2020")==True]

# Plotting based on date
import matplotlib.pyplot as plt
# Convert the 'date' column to a pandas DateTime format
barnet2020['Month'] = pd.to_datetime(barnet2020['Month'])

# Set the 'date' column as the DataFrame index
barnet2020.set_index('Month', inplace=True)

# Group the data by month and count the occurrences
monthly_counts = barnet2020.groupby(pd.Grouper(freq='M')).size()

# Create the plot
plt.plot(monthly_counts.index, monthly_counts.values)
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Monthly Data')
plt.xticks(rotation=45)
plt.show()

# Clustering with lat and long
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Prepare the input data for clustering
X = barnet_only[['Latitude', 'Longitude']]

num_clusters = 6

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

# Add the cluster labels back to the DataFrame
df['cluster'] = cluster_labels

# Plot the clusters
plt.scatter(df['latitude'], df['longitude'], c=df['cluster'])
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Cluster Visualization')
plt.show()