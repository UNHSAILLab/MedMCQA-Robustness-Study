"""Checkpointing utilities for experiment resume capability."""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExperimentCheckpoint:
    """Save and restore experiment progress for resume capability."""

    def __init__(
        self,
        experiment_name: str,
        checkpoint_dir: str = "outputs/checkpoints"
    ):
        """Initialize checkpoint manager.

        Args:
            experiment_name: Name of the experiment
            checkpoint_dir: Directory to store checkpoint files
        """
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{experiment_name}_checkpoint.json"
        )

    def save(
        self,
        processed_ids: List[str],
        partial_results: Dict[str, Any],
        metadata: Optional[Dict] = None
    ):
        """Save checkpoint to disk.

        Args:
            processed_ids: List of completed item IDs
            partial_results: Results collected so far
            metadata: Additional metadata to save
        """
        checkpoint_data = {
            "experiment_name": self.experiment_name,
            "processed_ids": processed_ids,
            "partial_results": partial_results,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "num_processed": len(processed_ids)
        }

        # Write to temp file first, then rename (atomic)
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        os.replace(temp_path, self.checkpoint_path)
        logger.debug(f"Checkpoint saved: {len(processed_ids)} items processed")

    def load(self) -> Tuple[List[str], Dict[str, Any], Dict]:
        """Load checkpoint from disk.

        Returns:
            Tuple of (processed_ids, partial_results, metadata)
        """
        if not self.exists():
            return [], {}, {}

        try:
            with open(self.checkpoint_path) as f:
                data = json.load(f)

            logger.info(
                f"Loaded checkpoint: {data['num_processed']} items, "
                f"saved at {data['timestamp']}"
            )

            return (
                data.get("processed_ids", []),
                data.get("partial_results", {}),
                data.get("metadata", {})
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return [], {}, {}

    def exists(self) -> bool:
        """Check if checkpoint file exists."""
        return os.path.exists(self.checkpoint_path)

    def clear(self):
        """Delete checkpoint file."""
        if self.exists():
            os.remove(self.checkpoint_path)
            logger.info(f"Checkpoint cleared: {self.checkpoint_path}")

    def get_remaining_items(
        self,
        all_items: List[Any],
        id_getter=lambda x: x.id
    ) -> List[Any]:
        """Get items that haven't been processed yet.

        Args:
            all_items: Complete list of items
            id_getter: Function to extract ID from item

        Returns:
            List of unprocessed items
        """
        processed_ids, _, _ = self.load()
        processed_set = set(processed_ids)

        remaining = [
            item for item in all_items
            if id_getter(item) not in processed_set
        ]

        if processed_ids:
            logger.info(
                f"Resuming: {len(processed_ids)} done, "
                f"{len(remaining)} remaining"
            )

        return remaining


class CheckpointManager:
    """Manage multiple experiment checkpoints."""

    def __init__(self, checkpoint_dir: str = "outputs/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def get_checkpoint(self, experiment_name: str) -> ExperimentCheckpoint:
        """Get checkpoint handler for an experiment."""
        return ExperimentCheckpoint(experiment_name, self.checkpoint_dir)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with their status."""
        checkpoints = []

        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith("_checkpoint.json"):
                path = os.path.join(self.checkpoint_dir, filename)
                try:
                    with open(path) as f:
                        data = json.load(f)
                    checkpoints.append({
                        "experiment_name": data.get("experiment_name"),
                        "num_processed": data.get("num_processed", 0),
                        "timestamp": data.get("timestamp"),
                        "file": filename
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

        return sorted(checkpoints, key=lambda x: x.get("timestamp", ""), reverse=True)

    def clear_all(self):
        """Clear all checkpoints."""
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith("_checkpoint.json"):
                os.remove(os.path.join(self.checkpoint_dir, filename))
        logger.info("All checkpoints cleared")
