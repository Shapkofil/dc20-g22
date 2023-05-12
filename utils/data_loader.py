#!/usr/bin/env python
# coding: utf-8

from abc import ABCMeta, abstractmethod

from functools import partial
from typing_extensions import override
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time

from pathlib import Path
import os

from tqdm import tqdm

import folium
from shapely import envelope
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from typing import Union, Any, List, Dict


class DataLoader():
    @staticmethod
    def slurp_request_to_df(url:str)->pd.DataFrame:
        # Rate limiting
        time.sleep(.1)

        response = requests.get(url)
        if response is None:
            raise RuntimeError(f"Error: GET call to {url} returns an empty responce")
        return pd.DataFrame(response.json())
            

    def __init__(self, data_path: Union[str, os.PathLike, Path]):
        self.df = pd.read_parquet(data_path)

    @abstractmethod
    def __getitem__(self, idx:int) -> Any:
        """
        Getting a data from the index of data
        """
        return self.df.iloc[idx]
        

class MetropolitanLoader(DataLoader):
    """
    MetropolitanLoader object. (down)Loads all of the metropolitan police data

    Parameters
    ----------
    data_path : str or Path
        The path to the directory containing the data files.
    cache_dir : str or Path, optional
        The path to the directory where the cached files will be stored.
        Defaults to "./cache".

    Returns
    -------
    None

    Notes
    -----
    If the cached files already exist, they will be loaded into memory instead
    of re-downloading and processing the data from scratch. Otherwise, the data
    will be downloaded and processed, and the resulting dataframes will be
    cached for future use.

    The MetropolitanLoader object stores the following attributes:
    - df: a pandas dataframe containing the crime data with an additional
      "ward" column indicating the name of the ward where the crime took place.
    - wards: a pandas dataframe containing the ward data, including the ward
      names, IDs, and boundary coordinates.
    - ward_envelopes: a dictionary mapping ward IDs to the corresponding
      bounding boxes, which are shapely Polygon objects.

    Examples
    --------
    >>> loader = MetropolitanLoader("data/", cache_dir="./cache")
    """


    def __init__(self,
                 data_path: Union[str, os.PathLike, Path],
                 cache_dir: Union[str, os.PathLike, Path] = "./cache"):

        data_path = Path(data_path)
        cache_dir = Path(cache_dir)
        if (cache_dir / Path("datainwards.parquet")).exists() and \
           (cache_dir / Path("wardboundries.parquet")).exists():
            self.df = pd.read_parquet(cache_dir / Path("datainwards.parquet"))
            self.wards = pd.read_parquet(cache_dir / Path("wardboundries.parquet"))
            self.ward_envelopes = self.compute_ward_envelopes()
            return 

        super().__init__(data_path)
        self.wards = DataLoader.slurp_request_to_df(
            "https://data.police.uk/api/metropolitan/neighbourhoods")
        self.wards.set_index("id", inplace=True)
        self.query_boundries()

        # Compute ward envelopes (bounding boxes)
        self.ward_envelopes = self.compute_ward_envelopes()

        # Compute ward per crime
        bar = tqdm.pandas(desc="Finding Wards...")
        self.df["ward"] = self.df.progress_apply(
            partial(MetropolitanLoader.find_ward, self), axis=1)

        if not cache_dir.exists():
            cache_dir.mkdir()
        self.df.to_parquet(cache_dir / Path("datainwards.parquet"))
        self.wards.to_parquet(cache_dir / Path("wardboundries.parquet"))


    def query_boundries(self):
        """
        Augment the wards data with the ward boundries
        """
        ids = self.wards.index.tolist()
        
        longi = []
        lat = []
        
        for id in tqdm(ids, desc="Retriving Ward Boundries"):
            # set the url
            df = DataLoader.slurp_request_to_df(
                f"https://data.police.uk/api/metropolitan/{id}/boundary")
            # append the longitude and latitude to the lists
            longi.append(df["longitude"].to_numpy().astype(float))
            lat.append(df["latitude"].to_numpy().astype(float))
            
        self.wards["longitude"] = longi
        self.wards["latitude"] = lat
        # new column boundaries which is a list of tuples of the longitude and latitude
        self.wards["boundaries"] = self.wards.apply(
            lambda x: list(zip(x["longitude"], x["latitude"])), axis=1)


    def compute_ward_envelopes(self)->List[Polygon]:
        """
        Computes ward envelopes (bounding boxes).
        This makes filtering crimes way faster
        """

        ward_envlps = []
        for bound in tqdm(self.wards["boundaries"], desc="Computing Ward Envelopes"):
            polygon = Polygon(bound)
            envpl = envelope(polygon)
            ward_envlps.append(envpl)
        return ward_envlps
 

    def find_ward(self, row:pd.Series) -> Union[str, float]:
        """
        Finds the ward where a crime took place. The function takes a row 
        from the df_burglary dataframe as input and returns the name of the ward
        where the crime took place.
        """
        point = Point(row["Longitude"], row["Latitude"])
        # check if the burglary point is within any of the ward buffers
        for idx, envlope in enumerate(self.ward_envelopes):
            if envlope.contains(point):
                # if the point is within the envelope, check if it's also within the ward polygon
                polygon = Polygon(self.wards.iloc[idx]["boundaries"])
                if polygon.contains(point):
                    return self.wards.iloc[idx]["name"]
        return np.nan



class BarnetLoader(MetropolitanLoader):
    """
    A subclass of MetropolitanLoader that loads crime data and ward boundaries
    for the London Borough of Barnet.

    Attributes
    ----------
    wards : pandas DataFrame
        A dataframe containing the ward data for the Barnet borough.
    df : pandas DataFrame
        A dataframe containing the crime data for the Barnet borough.

    Examples
    --------
    >>> loader = BarnetLoader("data/")
    """

    BARNET_WARDS = ["High Barnet",
                    "Underhill",
                    "Barnet Vale",
                    "East Barnet",
                    "Friern Barnet",
                    "Woodhouse",
                    "Whetstone",
                    "Brunswick Park",
                    "Totteridge and Woodside",
                    "Mill Hill",
                    "Cricklewood",
                    "Edgwarebury",
                    "Burnt Oak",
                    "Edgware",
                    "Colindale South",
                    "West Hendon",
                    "Colindale North",
                    "Hendon",
                    "West Finchley",
                    "East Finchley",
                    "Garden Suburb",
                    "Finchley Church End",
                    "Golders Green",
                    "Childs Hill"]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Filter barnet wards
        self.wards = pd.DataFrame(self.wards)[
            self.wards["name"].isin(BarnetLoader.BARNET_WARDS)]

        # Filter barnet crimes
        self.df = self.df[
            self.df["ward"].isin(BarnetLoader.BARNET_WARDS)]


if __name__ == "__main__":
    bl = BarnetLoader("../data/street.parquet")
    print(len(bl.df))
