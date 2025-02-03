import numpy as np
import pandas as pd
from scipy.spatial import distance

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
        self.cluster_centers_ = None
        self.inertia_ = 0

    def __str__(self) -> str:
        params = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "MyKMeans class: " + ", ".join(params)


    def _calculate_wcss(self, X: pd.DataFrame, labels: np.ndarray) -> float:
        """Вычисляет WCSS (Within-Cluster Sum of Squares) для данной кластеризации."""
        wcss = 0
        for cluster_idx in range(self.n_clusters):
            cluster_points = X[labels == cluster_idx]
            if not cluster_points.empty:  
              centroid = self.cluster_centers_[cluster_idx]
              for point in cluster_points.values:
                    wcss += distance.euclidean(point, centroid) ** 2
        return wcss

    def _assign_labels(self, X: pd.DataFrame) -> np.ndarray:
        """Назначает каждой точке ближайший центроид."""
        labels = np.zeros(len(X), dtype=int)
        for i, point in enumerate(X.values):
            distances = [distance.euclidean(point, centroid) for centroid in self.cluster_centers_]
            labels[i] = np.argmin(distances)
        return labels

    def _update_centroids(self, X: pd.DataFrame, labels: np.ndarray) -> None:
        """Пересчитывает координаты центроидов на основе текущего распределения точек."""
        for cluster_idx in range(self.n_clusters):
            cluster_points = X[labels == cluster_idx]
            if not cluster_points.empty:  
                self.cluster_centers_[cluster_idx] = cluster_points.mean().values

    def fit(self, X: pd.DataFrame) -> None:

        np.random.seed(seed=self.random_state)
        best_inertia = float('inf')
        best_centers = None


        for _ in range(self.n_init):
            centroids = []
            for _ in range(self.n_clusters):
                centroid = [np.random.uniform(X[col].min(), X[col].max()) for col in X.columns]
                centroids.append(centroid)
            self.cluster_centers_ = centroids


            for _ in range(self.max_iter):
              labels = self._assign_labels(X)
              self._update_centroids(X, labels)

            current_inertia = self._calculate_wcss(X, labels)


            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_centers = self.cluster_centers_

        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers

    def predict(self, X: pd.DataFrame):
        pred = []
        for i in range(X.shape[0]):
            d = float('inf')
            point = X.iloc[i]
            for j in range(self.n_clusters):
                centroid = self.cluster_centers_[j]
                if d > distance.euclidean(point, centroid):
                    d = distance.euclidean(point, centroid)
                    clust = j + 1
            pred.append(clust)

        return pred