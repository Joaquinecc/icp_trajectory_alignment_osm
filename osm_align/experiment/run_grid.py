import os
import sys
import time
import itertools
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_command(python_exec: str, script_path: str,
                  frame_id: str,
                  pose_segment_size: int,
                  icp_error_threshold: float,
                  trimming_ratio: float,
                  knn_neighbors: int,
                  valid_correspondence_threshold: float,
                  save_path_folder: str) -> list:
    return [
        python_exec,
        script_path,
        "--frame_id", str(frame_id),
        "--pose_segment_size", str(pose_segment_size),
        "--icp_error_threshold", str(icp_error_threshold),
        "--trimming_ratio", str(trimming_ratio),
        "--knn_neighbors", str(knn_neighbors),
        "--valid_correspondence_threshold", str(valid_correspondence_threshold),
        "--save_resuts_path", save_path_folder,
    ]


def run_one(python_exec: str, script_path: str, params: tuple, results_base_dir: str) -> tuple:
    (frame_id, pose_segment_size, icp_error_threshold, trimming_ratio, knn_neighbors, valid_corr) = params
    folder = f"{pose_segment_size}_{icp_error_threshold}_{trimming_ratio}_{knn_neighbors}_{valid_corr}"
    save_path_folder = os.path.join(results_base_dir, f"{frame_id}/{folder}/results.txt")
    if os.path.exists(save_path_folder):
        print(f"Skipping frame={frame_id} seg={pose_segment_size} icp={icp_error_threshold} trim={trimming_ratio} knn={knn_neighbors} valid={valid_corr} because it already exists", flush=True)
        return params, 0, 0

    cmd = build_command(
        python_exec,
        script_path,
        frame_id,
        pose_segment_size,
        icp_error_threshold,
        trimming_ratio,
        knn_neighbors,
        valid_corr,
        save_path_folder,
    )
    start = time.time()
    try:
        proc = subprocess.run(cmd, stdout=None, stderr=None)
        rc = proc.returncode
    except Exception as e:
        rc = -1
        print(f"ERR running {params}: {e}", flush=True)
    elapsed = time.time() - start
    print(
        f"Finished frame={frame_id} seg={pose_segment_size} icp={icp_error_threshold} trim={trimming_ratio} knn={knn_neighbors} valid={valid_corr} in {elapsed:.2f}s (rc={rc})",
        flush=True,
    )
    return params, rc, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run simple_test.py over a parameter grid using threads.")
    parser.add_argument("--max_workers", type=int, default=min(32, (os.cpu_count() or 8) + 4), help="Maximum concurrent threads")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of jobs (0 = no limit)")
    parser.add_argument("--results_base_dir", type=str, help="Base directory for saving results")
    args = parser.parse_args()

    # Parameter grids (as requested)
    # FRAME_IDS = ("07", "02", "08", "05", "06", "09", "10", "00")
    FRAME_IDS = ("01","04")
    POSE_HISTORY_SIZES = (20, 50, 100, 150, 200)
    ICP_ERROR_THRESHOLDS = (0.5, 1.0, 1.5, 2.0)
    TRIMMING_RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4)
    KNN_NEIGHBORS = (10, 20, 50, 100)
    VALID_CORRESPONDENCE_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)

    # Resolve script path
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "simple_test.py"))
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"simple_test.py not found at {script_path}")

    all_params = list(itertools.product(
        FRAME_IDS,
        POSE_HISTORY_SIZES,
        ICP_ERROR_THRESHOLDS,
        TRIMMING_RATIOS,
        KNN_NEIGHBORS,
        VALID_CORRESPONDENCE_THRESHOLDS,
    ))

    if args.limit and args.limit > 0:
        all_params = all_params[: args.limit]

    print(f"Total jobs: {len(all_params)} | max_workers={args.max_workers}", flush=True)

    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(run_one, args.python, script_path, params, args.results_base_dir) for params in all_params]
        for fut in as_completed(futures):
            results.append(fut.result())

    total_elapsed = time.time() - started
    n_ok = sum(1 for _, rc, _ in results if rc == 0)
    print(f"Completed {len(results)} jobs in {total_elapsed/60.0:.2f} min | success={n_ok} failures={len(results)-n_ok}", flush=True)


if __name__ == "__main__":
    main() 