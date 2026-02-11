#!/usr/bin/env python3
"""
Convenience script for multi-seed experiment runs.

Launches parallel seed runs across GPUs and aggregates results after
completion.

Usage:
    # Run option_order experiment with 5 seeds on 4 GPUs
    python scripts/run_multi_seed.py --experiment option_order --model 4b \
        --seeds 42,123,456,789,1337 --gpu-ids 0,1,2,3

    # Aggregate only (skip running, just re-aggregate existing results)
    python scripts/run_multi_seed.py --experiment option_order --model 4b \
        --seeds 42,123,456,789,1337 --aggregate-only --results-dir outputs/results
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 123, 456, 789, 1337]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_single_seed(experiment, model, seed, gpu_id, limit=None, config=None, output_dir="outputs/results"):
    """Run a single seed on a specific GPU. Returns a result dict."""
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"python scripts/run_experiment.py {experiment} "
        f"-m {model} --seed {seed} -o {output_dir}"
    )
    if limit:
        cmd += f" -l {limit}"
    if config:
        cmd += f" -c {config}"

    start_time = time.time()
    logger.info(f"[GPU {gpu_id}] Starting seed={seed}: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        elapsed = time.time() - start_time
        return {
            "seed": seed,
            "gpu_id": gpu_id,
            "success": result.returncode == 0,
            "elapsed_seconds": elapsed,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:] if result.stdout else "",
            "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        }
    except Exception as e:
        return {
            "seed": seed,
            "gpu_id": gpu_id,
            "success": False,
            "error": str(e),
        }


def aggregate_seed_results(results_dir, experiment, model, seeds):
    """Load individual seed result files and compute aggregated statistics."""
    from src.experiments.multi_seed_runner import MultiSeedRunner

    seed_results = []
    for seed in seeds:
        # Find result files matching this experiment/model/seed
        pattern = os.path.join(results_dir, f"{experiment}_{model}_*.json")
        candidates = glob.glob(pattern)

        for filepath in sorted(candidates, reverse=True):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                if data.get("config", {}).get("seed") == seed:
                    seed_results.append({
                        "metrics": data.get("metrics", {}),
                        "seed": seed,
                        "output_path": filepath,
                    })
                    logger.info(f"Found results for seed {seed}: {filepath}")
                    break
            except (json.JSONDecodeError, KeyError):
                continue

    if not seed_results:
        logger.warning("No seed result files found to aggregate.")
        return None

    logger.info(f"Aggregating {len(seed_results)} seed results")
    aggregated = MultiSeedRunner.aggregate_results(seed_results)

    # Save aggregated output
    output = {
        "experiment": experiment,
        "model": model,
        "seeds": seeds,
        "seeds_found": [r["seed"] for r in seed_results],
        "individual_files": [r["output_path"] for r in seed_results],
        "aggregated": aggregated,
        "timestamp": datetime.now().isoformat(),
    }

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multi_seed_agg_{experiment}_{model}_{run_id}.json"
    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Aggregated results saved to {path}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with multiple seeds and aggregate results"
    )

    parser.add_argument(
        "--experiment", "-e",
        required=True,
        help="Experiment name (e.g., option_order, prompt_ablation)",
    )

    parser.add_argument(
        "--model", "-m",
        default="4b",
        help="Model variant (default: 4b)",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=f"Comma-separated list of seeds (default: {','.join(str(s) for s in DEFAULT_SEEDS)})",
    )

    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (e.g., '0,1,2,3')",
    )

    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of items (for testing)",
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip running; only aggregate existing results",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run seeds sequentially instead of parallel",
    )

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]

    print(f"Experiment: {args.experiment}")
    print(f"Model: {args.model}")
    print(f"Seeds: {seeds}")
    print(f"GPUs: {gpu_ids}")

    if not args.aggregate_only:
        # Assign seeds to GPUs round-robin
        tasks = []
        for i, seed in enumerate(seeds):
            gpu = gpu_ids[i % len(gpu_ids)]
            tasks.append((args.experiment, args.model, seed, gpu, args.limit, args.config, args.output_dir))

        print(f"\nLaunching {len(tasks)} seed runs...")
        print(f"Start time: {datetime.now().isoformat()}\n")

        results = []

        if args.sequential:
            for task_args in tasks:
                result = run_single_seed(*task_args)
                results.append(result)
                status = "SUCCESS" if result["success"] else "FAILED"
                print(f"[{status}] seed={result['seed']} on GPU {result['gpu_id']} "
                      f"({result.get('elapsed_seconds', 0):.1f}s)")
        else:
            with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
                futures = {
                    executor.submit(run_single_seed, *task_args): task_args[2]
                    for task_args in tasks
                }
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        status = "SUCCESS" if result["success"] else "FAILED"
                        print(f"[{status}] seed={result['seed']} on GPU {result['gpu_id']} "
                              f"({result.get('elapsed_seconds', 0):.1f}s)")
                    except Exception as e:
                        print(f"[ERROR] seed={seed}: {e}")
                        results.append({"seed": seed, "success": False, "error": str(e)})

        # Summary
        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful
        print(f"\nCompleted: {successful}/{len(results)} successful, {failed} failed")

        if failed > 0:
            print("\nFailed seeds:")
            for r in results:
                if not r.get("success"):
                    print(f"  seed={r['seed']}: {r.get('error', r.get('stderr_tail', '')[:200])}")

    # Aggregate
    print("\nAggregating results...")
    agg = aggregate_seed_results(args.output_dir, args.experiment, args.model, seeds)

    if agg:
        print("\n" + "=" * 60)
        print("AGGREGATED RESULTS")
        print("=" * 60)
        for key, stats in agg.get("aggregated", {}).items():
            if isinstance(stats, dict) and "mean" in stats:
                print(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
                      f"(95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}])")
        print("=" * 60)


if __name__ == "__main__":
    main()
