import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

import argparse

class DataPrepper:
    def __init__(self,
                 input_path: str,
                 scaling: str):
        self.df = pd.read_parquet(input_path)

        onehot_df = self.df.iloc[
            :, [1] + list(range(64-23, 64))]
        self.month_embeds = onehot_df.groupby("Month").sum()

        if scaling == "robust":
            self.scaler = RobustScaler()
            self.scaler.fit(self.month_embeds)
        elif scaling == "standard":
            self.scaler = StandardScaler()
            self.scaler.fit(self.month_embeds)
        else:
            raise NotImplementedError(f"{scaling} scaling is not supported")

        self.normalized_df = pd.DataFrame(
            self.scaler.transform(self.month_embeds),
            columns=self.month_embeds.columns,
            index=self.month_embeds.index)

    def save(self, path):
        self.normalized_df.to_parquet(path)


def main(args):
    dp = DataPrepper(args["dataset_path"], args["scaling"])
    dp.save(args["output_path"])


def start():
    parser = argparse.ArgumentParser(
        description="LSTM Data Preperation and Feature extraction")
    parser.add_argument("--dataset_path", "-d", type=str, default="../../data/street.parquet",
                        help="Path to the linearmodel dataset")
    parser.add_argument("--output_path", type=str, help="Path to write lstm.parquet")
    parser.add_argument("--scaling", type=str,
                        choices=["robust", "standard", "minmax"],
                        help="Kin  of scaling")

    args = vars(parser.parse_args())
    main(args)


if __name__ == "__main__":
    start()
