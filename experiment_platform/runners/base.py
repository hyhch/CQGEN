"""Base runner interface for CQ generation methods."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class RunResult:
    """Result from a single CQ generation run."""
    method: str
    dataset: str
    generated_cqs: list
    metrics: dict = field(default_factory=dict)
    intermediate_logs: list = field(default_factory=list)
    duration_seconds: float = 0.0


class BaseRunner(ABC):
    """Abstract base class for CQ generation method runners."""

    name: str = "base"

    @abstractmethod
    def run(
        self,
        dataset_name: str,
        llm_config: dict,
        params: dict,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> RunResult:
        """Run CQ generation on the given dataset.

        Args:
            dataset_name: Name of the dataset (e.g. 'demcare').
            llm_config: Dict with keys: model, api_key, base_url, api_type.
            params: Method-specific parameters.
            progress_callback: Called with log messages for streaming UI.

        Returns:
            RunResult with generated CQs and metadata.
        """

    def _log(self, msg: str, callback: Optional[Callable[[str], None]] = None):
        """Log a message, calling the progress callback if provided."""
        print(msg)
        if callback:
            callback(msg)
