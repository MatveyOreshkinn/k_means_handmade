import numpy as np
import pandas as pd


class MyKMeans:
    def __init__(self,
                 n_clusters: int = 3,
                 max_iter: int = 10,
                 n_init: int = 3,
                 random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def __str__(self) -> str:
        return f'MyKMeans class: n_clusters={self.n_clusters}, max_iter={self.max_iter}, n_init={self.n_init}, random_state={self.random_state}'
