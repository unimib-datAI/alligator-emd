import multiprocessing as mp
from functools import partial
from typing import Tuple

from alligator.config import AlligatorConfig
from alligator.ml import MLWorker

from .BaseProcessor import BaseProcessor


class MLProcessor(BaseProcessor):
    processor_id = "ml-processor"

    def __init__(self, config: AlligatorConfig):
        self.config = config

    def process(self, feature):
        """Run the full ML pipeline: rank stage, compute frequencies, rerank stage."""
        print("Running ML from processor class")
        pool = mp.Pool(processes=self.config.worker.num_workers or 1)
        try:
            pool.map(
                partial(self._run_worker, stage="rank", global_frequencies=(None, None, None)),
                range(self.config.ml.num_ml_workers),
            )

            global_frequencies = feature.compute_global_frequencies(
                docs_to_process=self.config.feature.doc_percentage_type_features,
                random_sample=False,
            )

            pool.map(
                partial(self._run_worker, stage="rerank", global_frequencies=global_frequencies),
                range(self.config.ml.num_ml_workers),
            )
        finally:
            pool.close()
            pool.join()

    def _run_worker(self, rank: int, stage: str, global_frequencies: Tuple):
        """Create and run a single MLWorker for the given stage."""
        model_path = (
            self.config.ml.ranker_model_path
            if stage == "rank"
            else self.config.ml.reranker_model_path
        )
        max_candidates = -1 if stage == "rank" else self.config.retrieval.max_candidates_in_result
        ml_worker = MLWorker(
            rank,
            table_name=self.config.data.table_name or "default_table",
            dataset_name=self.config.data.dataset_name or "default_dataset",
            stage=stage,
            model_path=model_path,
            batch_size=self.config.ml.ml_worker_batch_size,
            max_candidates_in_result=max_candidates,
            top_n_cta_cpa_freq=self.config.feature.top_n_cta_cpa_freq,
            features=self.config.ml.selected_features,
            mongo_uri=self.config.database.mongo_uri or "mongodb://gator-mongodb:27017/",
            db_name=self.config.database.db_name or "alligator_db",
            input_collection=self.config.database.input_collection,
        )
        return ml_worker.run(global_frequencies=global_frequencies)
