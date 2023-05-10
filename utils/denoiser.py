import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.spatial.distance import cdist
from typing import Callable, Tuple, Generator

class StochasticDenoiser():
    """
    Generates wave function of crimes based on a distance to crime
    """

    DF_SAMPLE = lambda x: np.clip(1.0 - .1*x, 0.0, 1.0)

    def __init__(self,
                 crime_points:np.ndarray,
                 distance_function:Callable,
                 resolution:Tuple[int,int] = (256, 256)
                 ):

        self.crime_points = crime_points
        self.distance_function = distance_function
        self.res = resolution

    def local_wave_gen(self,
                       mesh_points:np.ndarray
                       )->Generator[np.ndarray, None, None]:
        for p in tqdm(self.crime_points):
            yield self.distance_function(
                cdist(p[np.newaxis, :], mesh_points).reshape(self.res))

    def generate_wave(self)->np.ndarray:
        # Define the grid of coordinates where we want to evaluate the Gaussian
        x, y = np.meshgrid(
            np.linspace(0, 1, self.res[0]),  # x coordinates
            np.linspace(0, 1, self.res[0]),  # y coordinates
        )

        mesh_points = np.column_stack((x.ravel(), y.ravel()))
        density = np.zeros(self.res)
        for wave in self.local_wave_gen(mesh_points):
            density += wave
        return density

    @staticmethod
    def plot_wave(wave:np.ndarray, points:np.ndarray=None, show=False)->None:
        plt.imshow(wave, cmap='gray')
        if not points is None:
            x, y = wave.shape
            plt.scatter(points[:, 0] * x, points[:, 1] * y, c='r')
        if show:
            plt.show()


if __name__ == "__main__":
    from data_loader import BarnetLoader
    bl = BarnetLoader("../data/street.parquet")
    sd = StochasticDenoiser(bl.df[["Longitude","Latitude"]].to_numpy(),
                            StochasticDenoiser.DF_SAMPLE)
    wave = sd.generate_wave()
    StochasticDenoiser.plot_wave(wave,
                                 # sd.crime_points,
                                 show=True)
