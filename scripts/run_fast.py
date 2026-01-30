#!/usr/bin/env python3
"""Fast parallel runner with optimized settings."""

import subprocess
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Configuration
GPU_IDS = [1, 2, 3, 4, 5, 6, 7]  # Skip GPU 0 (in use)
CONFIG = "configs/fast.yaml"

# Only run 4B model - 27B has parsing issues
EXPERIMENTS = [
    ("prompt_ablation", "4b"),
    ("option_order", "4b"),
    ("evidence_conditioning", "4b"),
    ("self_consistency", "4b"),
]


def run_experiment(gpu_id, experiment, model):
    """Run a single experiment on specified GPU."""
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py -e {experiment} -m {model} -c {CONFIG}"
    print(f"[GPU {gpu_id}] Starting: {experiment} ({model})")
    print(f"[GPU {gpu_id}] Command: {cmd}")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    return {
        "gpu": gpu_id,
        "experiment": experiment,
        "model": model,
        "success": result.returncode == 0,
        "stderr": result.stderr[-500:] if result.stderr else ""
    }


def main():
    print("=" * 60)
    print("FAST PARALLEL RUNNER")
    print("=" * 60)
    print(f"Config: {CONFIG}")
    print(f"GPUs: {GPU_IDS}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print()

    # Assign experiments to GPUs
    assignments = []
    for i, (exp, model) in enumerate(EXPERIMENTS):
        gpu = GPU_IDS[i % len(GPU_IDS)]
        assignments.append((gpu, exp, model))

    print("Assignments:")
    for gpu, exp, model in assignments:
        print(f"  GPU {gpu}: {exp} ({model})")
    print()

    print(f"Starting at {datetime.now().isoformat()}")
    print()

    # Run in parallel
    results = []
    with ProcessPoolExecutor(max_workers=len(GPU_IDS)) as executor:
        futures = {
            executor.submit(run_experiment, gpu, exp, model): (exp, model)
            for gpu, exp, model in assignments
        }

        for future in as_completed(futures):
            exp, model = futures[future]
            result = future.result()
            results.append(result)
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"[{status}] {exp} ({model}) on GPU {result['gpu']}")
            if not result["success"]:
                print(f"  Error: {result['stderr'][:200]}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success = sum(1 for r in results if r["success"])
    print(f"Successful: {success}/{len(results)}")
    print(f"Finished at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
