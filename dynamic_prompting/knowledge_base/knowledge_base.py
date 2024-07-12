import os

import numpy as np
from dynamic_prompting.embedings.config import EmbeddingsConfig
from dynamic_prompting.embedings.embeddings import Embeddings
from dynamic_prompting.knowledge_base.ann import NearestNeighbor
from dynamic_prompting.knowledge_base.config import KnowledgeBaseConfig
from dynamic_prompting.utils.utils import get_project_root


class KnowledgeBaseManagement(Embeddings, NearestNeighbor):
    """
    A class to manage a knowledge base for signal classification using embeddings and nearest neighbor search.
    """

    def __init__(self, embeddings_config: EmbeddingsConfig, kb_config: KnowledgeBaseConfig) -> None:
        """
        Initialize the KnowledgeBaseManagement class.

        :param embeddings_config: Configuration for the embeddings.
        :param kb_config: Configuration for the knowledge base.
        """
        super().__init__(embedding_config=embeddings_config)

        self.root_path = get_project_root()
        self.knowledge_base_config = kb_config
        kb_path = f'{self.root_path}/knowledge_bases'
        os.makedirs(kb_path, exist_ok=True)
        self.kb_file = f'{kb_path}/{self.knowledge_base_config.knowledge_base_name}.npy'

    def _data_loader(self):
        """
        Load dataset.

        Dataset should be in the following format:
        sample, label, context
        Example:
        this is a test, positive, test is an entity
        """
        pass

    def create_kb(self, texts: list):
        """
        Create the knowledge base from a list of texts.

        :param texts: List of texts to be embedded and stored in the knowledge base.
        """
        embeddings = self.get_embedding(texts)
        self._save_kb(embeddings)

    def _save_kb(self, embeddings: np.ndarray):
        """
        Save the embeddings to a file.

        :param embeddings: Embeddings to be saved.
        """
        with open(self.kb_file, 'wb') as f:
            np.save(f, embeddings)

    def update_kb(self, new_embeddings: list) -> np.ndarray:
        """
        Update the knowledge base with new embeddings.

        :param new_embeddings: List of new embeddings to be added.
        :return: Updated embeddings.
        """
        embeddings = self.load_kb()
        embeddings.append(new_embeddings)
        embeddings = np.array(embeddings)
        self._save_kb(embeddings)
        return embeddings

    def load_kb(self) -> list:
        """
        Load the embeddings from the knowledge base file.

        :return: List of embeddings.
        """
        with open(self.kb_file, 'rb') as f:
            embeddings = np.load(f)
        embeddings = embeddings.tolist()
        return embeddings

    def search_kb(self, query: str, embeddings: list, num_of_neighbours: int = 3) -> list:
        """
        Search the knowledge base for nearest neighbors to the query.

        :param query: Query string to be searched.
        :param embeddings: List of embeddings to search within.
        :param num_of_neighbours: Number of nearest neighbors to return.
        :return: Indices of the nearest neighbors.
        """
        query_embedding = self.get_embedding([query]).tolist()[
            0]
        indices = self.run_approximate_nearest_neighbor(
            embeddings=embeddings,
            query_embedding=query_embedding
        )

        indices = indices[:num_of_neighbours]
        indices = indices.tolist()
        return indices
