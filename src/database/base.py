from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(ABC):
    """Abstract base class for vector database storage."""

    @abstractmethod
    def insert(self, vector: np.ndarray, metadata: dict[str, Any]) -> str:
        """
        Insert a vector with metadata into the database.

        Args:
            vector: The embedding vector.
            metadata: Metadata associated with the vector.

        Returns:
            The unique identifier (UUID) of the inserted point.
        """
        pass

    @abstractmethod
    def search(
        self, query: np.ndarray, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Search for nearest neighbors of a query vector.

        Args:
            query: The query embedding vector.
            top_k: Number of nearest neighbors to return.
            filters: Optional metadata filters.

        Returns:
            A list of SearchResult objects.
        """
        pass

    @abstractmethod
    def create_collection(self, dimension: int, distance: str = "Cosine") -> None:
        """Create the vector collection if it doesn't exist."""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the vector collection."""
        pass

    @abstractmethod
    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the vector collection."""
        pass
