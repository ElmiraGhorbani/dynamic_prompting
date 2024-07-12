from typing import List

import numpy as np
from scipy import spatial


class NearestNeighbor:
    """
    A class to perform nearest neighbor search using various distance metrics.
    """

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        Compute the cosine similarity between two vectors.

        Args:
            a (List[float]): First vector.
            b (List[float]): Second vector.

        Returns:
            float: Cosine similarity between vector a and vector b.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def distances_from_embeddings(
        query_embedding: List[float],
        embeddings: List[List[float]],
        distance_metric: str = "cosine"
    ) -> List[float]:
        """
        Return the distances between a query embedding and a list of embeddings.

        Args:
            query_embedding (List[float]): The query embedding vector.
            embeddings (List[List[float]]): List of embedding vectors to compare against.
            distance_metric (str): The distance metric to use ('cosine', 'L1', 'L2', 'Linf').

        Returns:
            List[float]: List of distances.
        """
        distance_metrics = {
            "cosine": spatial.distance.cosine,
            "L1": spatial.distance.cityblock,
            "L2": spatial.distance.euclidean,
            "Linf": spatial.distance.chebyshev,
        }
        distances = [
            distance_metrics[distance_metric](query_embedding, embedding)
            for embedding in embeddings
        ]
        return distances

    @staticmethod
    def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
        """
        Return a list of indices of nearest neighbors from a list of distances.

        Args:
            distances (List[float]): List of distances.

        Returns:
            np.ndarray: Indices of nearest neighbors sorted by distance.
        """
        return np.argsort(distances)

    def run_approximate_nearest_neighbor(
        self,
        embeddings: List[List[float]],
        query_embedding: List[float],
        distance_metric: str = "cosine"
    ) -> np.ndarray:
        """
        Perform approximate nearest neighbor search and return indices of nearest neighbors.

        Args:
            embeddings (List[List[float]]): List of embedding vectors.
            query_embedding (List[float]): The query embedding vector.
            distance_metric (str): The distance metric to use ('cosine', 'L1', 'L2', 'Linf').

        Returns:
            np.ndarray: Indices of nearest neighbors sorted by distance.
        """
        # Get distances between the query embedding and other embeddings
        distances = self.distances_from_embeddings(
            query_embedding, embeddings, distance_metric)

        # Get indices of nearest neighbors
        indices_of_nearest_neighbors = self.indices_of_nearest_neighbors_from_distances(
            distances)
        return indices_of_nearest_neighbors
