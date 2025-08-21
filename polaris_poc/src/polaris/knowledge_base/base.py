"""
Abstract Base Class for a Knowledge Base storage provider.

This defines the contract that all KB storage implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import KBEntry, KBQuery, KBResponse


class BaseKnowledgeBase(ABC):
    """
    Interface for a knowledge base that stores and retrieves information
    about the managing system, its goals, decisions, and observations.
    """

    @abstractmethod
    def store(self, entry: KBEntry) -> bool:
        """
        Stores or updates an entry in the knowledge base.

        Args:
            entry: The KBEntry object to store.

        Returns:
            True if storage was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get(self, entry_id: str) -> Optional[KBEntry]:
        """
        Retrieves a single entry by its unique ID.

        Args:
            entry_id: The ID of the entry to retrieve.

        Returns:
            The KBEntry object if found, otherwise None.
        """
        pass

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """
        Deletes an entry from the knowledge base.

        Args:
            entry_id: The ID of the entry to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        pass

    @abstractmethod
    def query(self, query: KBQuery) -> KBResponse:
        """
        Executes a query against the knowledge base.

        Args:
            query: The KBQuery object describing the search.

        Returns:
            A KBResponse object containing the results and metadata.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Removes all entries and indexes from the knowledge base.
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves statistics about the knowledge base content.

        Returns:
            A dictionary containing statistics like entry counts, index sizes, etc.
        """
        pass
