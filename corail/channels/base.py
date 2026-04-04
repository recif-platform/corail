"""Channel abstract base class."""

from abc import ABC, abstractmethod

from corail.config import Settings
from corail.core.pipeline import Pipeline


class Channel(ABC):
    """Base class for I/O channels. Channel is the OUTER layer that starts the server."""

    def __init__(self, pipeline: Pipeline, settings: Settings) -> None:
        self.pipeline = pipeline
        self.settings = settings

    @abstractmethod
    def start(self) -> None:
        """Start the channel (blocks until shutdown)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the channel gracefully."""
        ...
