import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Specify the path to your Parquet file
file_path = "../data/linearmodel.parquet"

# Read the Parquet file into a pandas DataFrame
df = pd.read_parquet(file_path)

onehot_df = df.iloc[:, [1] + list(range(64-23,64))]
month_embeds = onehot_df.groupby("Month").sum()
month_totals = month_embeds.sum(axis=1)
month_pdf = month_embeds.apply(lambda row: row / month_totals[row.name], axis=1) * 2.0 - 1

month_pdf.to_parquet( "../data/lstm.parquet")

scaler = StandardScaler()
normalized_df = scaler.fit_transform(month_embeds)
normalized_df = pd.DataFrame(normalized_df, columns=month_embeds.columns)

normalized_df.to_parquet( "../data/lstm.parquet")


dat = np.reshape(np.array([i for i in range(400)]), (20,20))

idx = np.array([i for i in range(20)])
np.random.shuffle(idx)
idx = np.reshape(idx, (-1, 5))
nplag = lambda x, y: np.stack([x - y + i for i in range(y + 1)], axis=-1)
