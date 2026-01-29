"""Base experiment runner class."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime
import yaml
import logging
from tqdm import tqdm

from ..utils.caching import ResponseCache
from ..utils.checkpointing import ExperimentCheckpoint
from ..data.schemas import ModelResponse

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Abstract base class for all experiments."""

    def __init__(
        self,
        model,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        output_dir: str = "outputs/results",
        use_cache: bool = True,
        use_checkpoint: bool = True
    ):
        """Initialize experiment.

        Args:
            model: Model instance (must have generate method)
            config: Configuration dict
            config_path: Path to config file (used if config not provided)
            output_dir: Directory for results
            use_cache: Whether to use response caching
            use_checkpoint: Whether to use checkpointing
        """
        self.model = model

        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            self.config = {}

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup caching
        self.use_cache = use_cache
        if use_cache:
            cache_dir = self.config.get('paths', {}).get('cache_dir', 'outputs/cache')
            self.cache = ResponseCache(os.path.join(cache_dir, 'responses.db'))
        else:
            self.cache = None

        # Setup checkpointing
        self.use_checkpoint = use_checkpoint
        if use_checkpoint:
            checkpoint_dir = self.config.get('paths', {}).get('checkpoints_dir', 'outputs/checkpoints')
            self.checkpoint = ExperimentCheckpoint(
                f"{self.name}_{self.model.name}",
                checkpoint_dir
            )
        else:
            self.checkpoint = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Experiment name for logging and file naming."""
        pass

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the experiment and return results."""
        pass

    @abstractmethod
    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze raw results and compute metrics."""
        pass

    def run_inference(
        self,
        items: List[Any],
        prompt_builder,
        generation_config: Optional[Dict] = None,
        parse_fn=None,
        show_progress: bool = True
    ) -> List[ModelResponse]:
        """Run inference on a list of items with caching.

        Args:
            items: List of data items
            prompt_builder: Function to build prompt from item
            generation_config: Generation parameters
            parse_fn: Function to parse model output
            show_progress: Whether to show progress bar

        Returns:
            List of ModelResponse objects
        """
        generation_config = generation_config or self.config.get('generation', {})
        results = []
        pending_items = []
        pending_prompts = []

        batch_size = self.config.get('inference', {}).get('batch_size', 4)

        # Check cache and collect pending items
        iterator = tqdm(items, desc=f"Processing {self.name}") if show_progress else items

        for item in iterator:
            prompt = prompt_builder(item)

            # Check cache
            if self.use_cache and self.cache is not None:
                cache_key = self.cache.make_key(
                    item.id, self.model.name, prompt, generation_config
                )
                cached = self.cache.get(cache_key)

                if cached is not None:
                    response = ModelResponse(
                        item_id=item.id,
                        prompt=prompt,
                        raw_output=cached['raw_output'],
                        parsed_answer=cached.get('parsed_answer'),
                        generation_time_ms=cached.get('generation_time_ms', 0),
                        metadata=cached.get('metadata', {})
                    )
                    results.append((item, response))
                    continue

            pending_items.append(item)
            pending_prompts.append(prompt)

        # Process pending items in batches
        if pending_items:
            logger.info(f"Processing {len(pending_items)} items (cache hits: {len(results)})")

            for i in tqdm(range(0, len(pending_items), batch_size),
                         desc="Batches", disable=not show_progress):
                batch_items = pending_items[i:i + batch_size]
                batch_prompts = pending_prompts[i:i + batch_size]

                # Generate
                import time
                start = time.time()
                outputs = self.model.generate(batch_prompts, **generation_config)
                elapsed_ms = (time.time() - start) * 1000

                # Process outputs
                for item, prompt, output in zip(batch_items, batch_prompts, outputs):
                    parsed = parse_fn(output) if parse_fn else None

                    response = ModelResponse(
                        item_id=item.id,
                        prompt=prompt,
                        raw_output=output,
                        parsed_answer=parsed,
                        generation_time_ms=elapsed_ms / len(batch_items)
                    )

                    # Cache result
                    if self.use_cache and self.cache is not None:
                        cache_key = self.cache.make_key(
                            item.id, self.model.name, prompt, generation_config
                        )
                        self.cache.set(
                            cache_key,
                            {
                                'raw_output': output,
                                'parsed_answer': parsed,
                                'generation_time_ms': response.generation_time_ms
                            },
                            model_name=self.model.name,
                            experiment=self.name
                        )

                    results.append((item, response))

                # Checkpoint periodically
                checkpoint_interval = self.config.get('inference', {}).get('checkpoint_interval', 100)
                if self.use_checkpoint and self.checkpoint and i % checkpoint_interval == 0:
                    self._save_checkpoint(results)

        # Sort by original order and return responses
        results.sort(key=lambda x: items.index(x[0]) if x[0] in items else 0)
        return [r[1] for r in results]

    def _save_checkpoint(self, results: List):
        """Save checkpoint with current progress."""
        processed_ids = [item.id for item, _ in results]
        partial_results = {
            item.id: response.model_dump()
            for item, response in results
        }
        self.checkpoint.save(processed_ids, partial_results)

    def save_results(
        self,
        results: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> str:
        """Save results and metrics to disk.

        Args:
            results: Raw experiment results
            metrics: Computed metrics

        Returns:
            Path to saved file
        """
        output = {
            'experiment': self.name,
            'model': self.model.name,
            'run_id': self.run_id,
            'config': self.config,
            'results': results,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        filename = f"{self.name}_{self.model.name}_{self.run_id}.json"
        path = os.path.join(self.output_dir, filename)

        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Results saved to {path}")
        return path

    def _load_config(self, path: str) -> Dict:
        """Load configuration from file."""
        with open(path) as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def full_run(self) -> Dict[str, Any]:
        """Execute full experiment pipeline: run -> analyze -> save."""
        logger.info(f"Starting experiment: {self.name}")
        logger.info(f"Model: {self.model.name}")

        # Run experiment
        results = self.run()

        # Analyze results
        metrics = self.analyze(results)

        # Save
        output_path = self.save_results(results, metrics)

        # Clear checkpoint on successful completion
        if self.checkpoint:
            self.checkpoint.clear()

        return {
            'results': results,
            'metrics': metrics,
            'output_path': output_path
        }
