"""Multi-seed experiment runner for statistical robustness."""

import copy
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Type

import numpy as np

from .base import BaseExperiment

logger = logging.getLogger(__name__)


class MultiSeedRunner:
    """Run an experiment across multiple random seeds and aggregate results.

    Enables reporting mean, std, 95% CI, min, and max for accuracy and
    flip_rate across randomized perturbations.
    """

    def __init__(
        self,
        experiment_class: Type[BaseExperiment],
        model,
        config: Dict[str, Any],
        seeds: List[int],
        output_dir: str = "outputs/results",
    ):
        self.experiment_class = experiment_class
        self.model = model
        self.base_config = config
        self.seeds = seeds
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Run experiment once per seed, collect and aggregate results."""
        seed_results: List[Dict[str, Any]] = []

        for i, seed in enumerate(self.seeds):
            logger.info(
                f"Multi-seed run {i + 1}/{len(self.seeds)} with seed={seed}"
            )
            config = copy.deepcopy(self.base_config)
            config["seed"] = seed

            experiment = self.experiment_class(
                model=self.model,
                config=config,
                output_dir=self.output_dir,
            )

            result = experiment.full_run()
            result["seed"] = seed
            seed_results.append(result)

        aggregated = self.aggregate_results(seed_results)

        # Save combined output
        combined = {
            "experiment": self.experiment_class.__name__,
            "model": self.model.name,
            "seeds": self.seeds,
            "individual_results": seed_results,
            "aggregated": aggregated,
            "timestamp": datetime.now().isoformat(),
        }
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"multi_seed_{self.experiment_class.__name__}"
            f"_{self.model.name}_{run_id}.json"
        )
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(combined, f, indent=2, default=str)

        logger.info(f"Multi-seed results saved to {path}")

        return combined

    @staticmethod
    def aggregate_results(
        seed_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute aggregated statistics across seed runs.

        For each metric key found in the seed results (accuracy, flip_rate,
        and any nested perturbation metrics), computes:
          - mean, std, 95% CI (lower, upper), min, max

        Args:
            seed_results: List of result dicts, each from experiment.full_run()

        Returns:
            Dict with aggregated statistics.
        """
        if not seed_results:
            return {}

        all_metrics = [r["metrics"] for r in seed_results]

        aggregated: Dict[str, Any] = {}

        # Collect top-level scalar metrics across seeds
        aggregated.update(
            MultiSeedRunner._aggregate_metric_dict(all_metrics)
        )

        # Handle nested 'perturbations' dict (e.g. from exp2_option_order)
        if "perturbations" in all_metrics[0]:
            pert_keys = all_metrics[0]["perturbations"].keys()
            aggregated["perturbations"] = {}
            for pert_key in pert_keys:
                pert_dicts = [
                    m["perturbations"][pert_key]
                    for m in all_metrics
                    if pert_key in m.get("perturbations", {})
                ]
                aggregated["perturbations"][pert_key] = (
                    MultiSeedRunner._aggregate_metric_dict(pert_dicts)
                )

        # Handle nested 'summary' dict
        if "summary" in all_metrics[0]:
            summary_dicts = [
                m["summary"] for m in all_metrics if "summary" in m
            ]
            aggregated["summary"] = (
                MultiSeedRunner._aggregate_metric_dict(summary_dicts)
            )

        # Handle 'original' dict (e.g. from exp2)
        if "original" in all_metrics[0] and isinstance(
            all_metrics[0]["original"], dict
        ):
            orig_dicts = [
                m["original"] for m in all_metrics if "original" in m
            ]
            aggregated["original"] = (
                MultiSeedRunner._aggregate_metric_dict(orig_dicts)
            )

        return aggregated

    @staticmethod
    def _aggregate_metric_dict(
        metric_dicts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate scalar values from a list of metric dicts.

        For each key whose value is a float or int, compute summary stats.
        Non-numeric values are skipped.
        """
        if not metric_dicts:
            return {}

        result: Dict[str, Any] = {}
        for key in metric_dicts[0]:
            values = []
            for d in metric_dicts:
                v = d.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values.append(float(v))

            if len(values) == len(metric_dicts):
                result[key] = MultiSeedRunner._compute_stats(values)

        return result

    @staticmethod
    def _compute_stats(values: List[float]) -> Dict[str, float]:
        """Compute mean, std, 95% CI, min, max for a list of values."""
        arr = np.array(values)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        # 95% CI using t-distribution approximation for small n
        if n > 1:
            from scipy import stats as sp_stats

            t_crit = sp_stats.t.ppf(0.975, df=n - 1)
            margin = t_crit * std / np.sqrt(n)
        else:
            margin = 0.0

        return {
            "mean": mean,
            "std": std,
            "ci_95_lower": mean - margin,
            "ci_95_upper": mean + margin,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n_seeds": n,
        }
