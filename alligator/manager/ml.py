from typing import Tuple

from alligator.config import AlligatorConfig
from alligator.feature import Feature
from alligator.log import get_logger
from alligator.manager.processors.BaseProcessor import BaseProcessor
import alligator.manager.processors  # noqa: F401 - ensures all processors are registered


class MLManager:
    """Manages machine learning pipeline for ranking and reranking."""

    def __init__(self, config: AlligatorConfig):
        self.config = config
        self.logger = get_logger("ml_manager")

    def run_ml_pipeline(self, feature: Feature, processor_id: str = "ml-processor") -> None:
        """Run the complete ML pipeline via the registered processor."""
        processor = BaseProcessor.registry[processor_id](self.config)
        processor.process(feature)

    def _ml_worker(
        self, rank: int, stage: str, global_frequencies: Tuple, processor_id: str = "ml-processor"
    ):
        """For backward compatibility — delegates to the processor's worker."""
        processor = BaseProcessor.registry[processor_id](self.config)
        return processor._run_worker(rank, stage, global_frequencies)
