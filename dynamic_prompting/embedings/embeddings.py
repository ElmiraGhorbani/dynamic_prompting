import os

import numpy as np
from dynamic_prompting.embedings.config import EmbeddingsConfig
from dynamic_prompting.utils.utils import get_project_root
from sentence_transformers import SentenceTransformer


class Embeddings:
    def __init__(self, embedding_config: EmbeddingsConfig):
        """
        Initialize Embeddings object.

        Args:
        - model_name (str): Name of the sentence embeddings model.
        - local_files (bool): Load models from local directory or download from huggingface
        - trust_remote_code (bool): Whether to trust remote code loaded from the model (default: False).
        - max_dimension (int): Maximum number of dimensions to truncate embeddings to (default: None).
        - epsilon (float): Small value added for numerical stability (default: 1e-5).
        """
        super().__init__()
        self.config = embedding_config

        # Get the root path of the project
        self.root_path = get_project_root()

        # Load the SentenceTransformer model
        self.model = self.loader()

    def loader(self):
        """
        Load the SentenceTransformer model from the specified path.

        Returns:
        - model (SentenceTransformer): Loaded SentenceTransformer model.
        """
        model_path = ''

        if self.config.local_files:
            model_path = f'{self.root_path}/models/{self.config.model_name}'

        if not os.path.exists(model_path):
            if 'nomic' in self.config.model_name:
                model_path = 'nomic-ai/nomic-embed-text-v1.5'
            elif 'bge' in self.config.model_name:
                model_path = 'BAAI/bge-large-zh-v1.5'
            elif 'mxbai' in self.config.model_name:
                model_path = 'mixedbread-ai/mxbai-embed-large-v1'

        model = SentenceTransformer(
            model_path,
            truncate_dim=self.config.max_dimension,
            trust_remote_code=self.config.trust_remote_code
        )

        return model

    def post_process_embeddings(self, embeddings):
        """
        Perform post-processing on embeddings:
        1. Standardize by subtracting mean and dividing by standard deviation.
        2. Truncate embeddings to specified max_dimension.
        3. Normalize embeddings using L2 normalization.

        Args:
        - embeddings (np.ndarray): Array of sentence embeddings.

        Returns:
        - processed_embeddings (np.ndarray): Processed embeddings after standardization and normalization.
        """
        mean = np.mean(embeddings, axis=1, keepdims=True)
        std = np.std(embeddings, axis=1, keepdims=True)

        # Adding epsilon for numerical stability
        embeddings = (embeddings - mean) / (std + self.config.epsilon)

        # Slicing to max_dimension
        embeddings = embeddings[:, :self.config.max_dimension]

        # L2 normalization along rows
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Adding epsilon for numerical stability
        embeddings = embeddings / (norms + self.config.epsilon)

        return embeddings

    def get_embedding(self, sentences: list):
        """
        Get processed embeddings for a list of sentences.

        Args:
        - sentences (list): List of sentences to encode.

        Returns:
        - embeddings (np.ndarray): Processed embeddings for the input sentences.
        """
        embeddings = self.model.encode(sentences)
        embeddings = self.post_process_embeddings(embeddings)
        return embeddings
