#!/usr/bin/env python3
"""
Parallel experiment runner for multi-GPU setup.

Distributes experiments across available GPUs for maximum throughput.

Usage:
    # Run all experiments across 8 GPUs
    python scripts/run_parallel.py --gpus 8

    # Run specific experiments
    python scripts/run_parallel.py --gpus 8 --experiments exp1 exp2

    # Run with specific models
    python scripts/run_parallel.py --gpus 8 --models 4b 27b
"""

import argparse
import subprocess
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional
import time
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Task:
    """A single experiment task."""
    name: str
    experiment: str
    model: str
    gpu_id: int
    limit: Optional[int] = None
    extra_args: str = ""

    def to_command(self) -> str:
        """Generate the command to run this task."""
        cmd = f"CUDA_VISIBLE_DEVICES={self.gpu_id} python main.py -e {self.experiment} -m {self.model}"
        if self.limit:
            cmd += f" -l {self.limit}"
        if self.extra_args:
            cmd += f" {self.extra_args}"
        return cmd


# Define all experiment configurations
EXPERIMENTS = {
    "exp1": {
        "name": "prompt_ablation",
        "description": "Prompt recipe ablation on MedMCQA",
        "dataset": "MedMCQA (4,183 items)",
        "estimated_time_4b": "3-4 hours",
        "estimated_time_27b": "8-10 hours"
    },
    "exp2": {
        "name": "option_order",
        "description": "Option order sensitivity on MedMCQA",
        "dataset": "MedMCQA (4,183 items x 5 perturbations)",
        "estimated_time_4b": "4-5 hours",
        "estimated_time_27b": "10-12 hours"
    },
    "exp3": {
        "name": "evidence_conditioning",
        "description": "Evidence conditioning on PubMedQA",
        "dataset": "PubMedQA (1,000 items x 6 conditions)",
        "estimated_time_4b": "1-2 hours",
        "estimated_time_27b": "3-4 hours"
    },
    "exp4": {
        "name": "self_consistency",
        "description": "Self-consistency voting on both datasets",
        "dataset": "MedMCQA + PubMedQA (multiple samples per item)",
        "estimated_time_4b": "6-8 hours",
        "estimated_time_27b": "15-20 hours"
    }
}

MODELS = {
    "4b": {
        "name": "MedGemma 4B",
        "quantization": None,
        "vram_required": "10GB"
    },
    "27b": {
        "name": "MedGemma 27B (4-bit)",
        "quantization": "4bit",
        "vram_required": "15GB"
    }
}


def run_task(task: Task) -> dict:
    """Run a single task and return results."""
    start_time = time.time()
    cmd = task.to_command()

    print(f"[GPU {task.gpu_id}] Starting: {task.name}")
    print(f"[GPU {task.gpu_id}] Command: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        elapsed = time.time() - start_time
        success = result.returncode == 0

        return {
            "task": task.name,
            "gpu_id": task.gpu_id,
            "success": success,
            "elapsed_seconds": elapsed,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:] if result.stdout else "",
            "stderr_tail": result.stderr[-2000:] if result.stderr else ""
        }

    except Exception as e:
        return {
            "task": task.name,
            "gpu_id": task.gpu_id,
            "success": False,
            "error": str(e)
        }


def create_task_schedule(
    experiments: List[str],
    models: List[str],
    num_gpus: int,
    limit: Optional[int] = None
) -> List[Task]:
    """Create a list of tasks distributed across GPUs."""
    tasks = []
    gpu_idx = 0

    for exp_key in experiments:
        if exp_key not in EXPERIMENTS:
            print(f"Warning: Unknown experiment {exp_key}, skipping")
            continue

        exp_name = EXPERIMENTS[exp_key]["name"]

        for model in models:
            if model not in MODELS:
                print(f"Warning: Unknown model {model}, skipping")
                continue

            task = Task(
                name=f"{exp_key}_{model}",
                experiment=exp_name,
                model=model,
                gpu_id=gpu_idx % num_gpus,
                limit=limit
            )
            tasks.append(task)
            gpu_idx += 1

    return tasks


def print_task_plan(tasks: List[Task]):
    """Print the execution plan."""
    print("\n" + "=" * 70)
    print("EXECUTION PLAN")
    print("=" * 70)

    by_gpu = {}
    for task in tasks:
        by_gpu.setdefault(task.gpu_id, []).append(task)

    for gpu_id in sorted(by_gpu.keys()):
        print(f"\nGPU {gpu_id}:")
        for task in by_gpu[gpu_id]:
            exp_info = EXPERIMENTS.get(task.experiment.replace("_", ""), {})
            print(f"  - {task.name}: {task.experiment} with {task.model}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments in parallel across multiple GPUs"
    )

    parser.add_argument(
        "--gpus", "-g",
        type=int,
        default=8,
        help="Number of GPUs available (default: 8)"
    )

    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default=["all"],
        help="Experiments to run"
    )

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Models to use"
    )

    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit items per experiment (for testing)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing"
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run tasks sequentially instead of parallel"
    )

    args = parser.parse_args()

    # Expand 'all' options
    experiments = list(EXPERIMENTS.keys()) if "all" in args.experiments else args.experiments
    models = list(MODELS.keys()) if "all" in args.models else args.models

    # Create task schedule
    tasks = create_task_schedule(experiments, models, args.gpus, args.limit)

    if not tasks:
        print("No tasks to run!")
        return

    print_task_plan(tasks)

    if args.dry_run:
        print("\nDry run - not executing tasks")
        return

    print(f"\nStarting {len(tasks)} tasks across {args.gpus} GPUs...")
    print(f"Start time: {datetime.now().isoformat()}\n")

    results = []

    if args.sequential:
        # Run sequentially
        for task in tasks:
            result = run_task(task)
            results.append(result)
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"[{status}] {task.name} completed in {result.get('elapsed_seconds', 0):.1f}s")
    else:
        # Run in parallel (one task per GPU at a time, cycling through)
        # Group tasks by GPU
        by_gpu = {}
        for task in tasks:
            by_gpu.setdefault(task.gpu_id, []).append(task)

        # Run tasks for each GPU in parallel
        with ProcessPoolExecutor(max_workers=args.gpus) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(run_task, task)
                futures.append((task, future))

            for task, future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    status = "SUCCESS" if result["success"] else "FAILED"
                    print(f"[{status}] {task.name} completed in {result.get('elapsed_seconds', 0):.1f}s")
                except Exception as e:
                    print(f"[ERROR] {task.name}: {e}")
                    results.append({"task": task.name, "success": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful

    print(f"Total tasks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed tasks:")
        for r in results:
            if not r.get("success"):
                print(f"  - {r['task']}: {r.get('error', r.get('stderr_tail', 'Unknown error')[:200])}")

    # Save results
    results_file = f"outputs/parallel_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("outputs", exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
