"""
Benchmark parallel vs serial implementation
Demonstrates speedup achieved
"""

import time
import multiprocessing as mp

from parallel_face_search import ParallelFaceRecognize
from serial_face_search import serial_face_recognition
from utils import format_time


def run_serial(known_image_path, folder_path, find_all=True):
    """
    Run serial face recognition

    Returns:
        matches, time_taken
    """
    print("Running serial implementation...")

    matches, time_taken = serial_face_recognition(
        known_image_path,
        folder_path,
        find_all=find_all,
        save_results=False
    )

    print(f"\nSerial Results:")
    print(f"Time taken: {time_taken:.2f}s")
    print(f"Matches found: {len(matches)}")

    return matches, time_taken


def run_parallel(known_image_path, folder_path, num_workers, find_all=True):
    """
    Run parallel face recognition

    Returns:
        matches, time_taken
    """
    print(f"\nRunning parallel implementation...")

    start_time = time.time()

    recognizer = ParallelFaceRecognize(known_image_path, num_workers=num_workers)
    recognizer.load_known_face()

    matches = recognizer.search_parallel(folder_path, find_all=find_all)

    time_taken = time.time() - start_time

    print(f"\nParallel Results:")
    print(f"Time taken: {time_taken:.2f}s")
    print(f"Matches found: {len(matches)}")

    return matches, time_taken, recognizer.num_workers


def compare_performance(find_all=True):
    """Compare serial vs parallel performance"""
    print("-----------------------------------------------------------")
    print("        Performance comparison: Serial vs Parallel")
    print("-----------------------------------------------------------\n")

    known_image = "dataset/known_woman.jpg"
    image_folder = "dataset/imageset/"

    # Serial
    serial_matches, serial_time = run_serial(known_image, image_folder, find_all)

    # Parallel (auto-detect workers)
    parallel_matches, parallel_time, actual_workers = run_parallel(
        known_image,
        image_folder,
        num_workers=None,
        find_all=find_all
    )

    print("\nAnalysis:")
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    efficiency = (speedup / actual_workers) * 100 if actual_workers > 0 else 0

    print(f"Serial Time:     {format_time(serial_time)}")
    print(f"Parallel Time:   {format_time(parallel_time)}")
    print(f"Speedup:         {speedup:.2f}x")
    print(f"Efficiency:      {efficiency:.1f}%")
    print(f"Workers Used:    {actual_workers}")

    print("\nInterpretation:")
    if efficiency >= 80:
        print(f"Excellent efficiency ({efficiency:.1f}%)")
    elif efficiency >= 60:
        print(f"Good efficiency ({efficiency:.1f}%)")
    elif efficiency >= 40:
        print(f"Moderate efficiency ({efficiency:.1f}%)")
    else:
        print(f"Low efficiency ({efficiency:.1f}%)")

    # Correctness check
    if set(serial_matches) == set(parallel_matches):
        print("\nResults match! Parallel implementation is correct.")
    else:
        print("\nResults differ between serial and parallel!")
        print(f"Serial found:   {len(serial_matches)} matches")
        print(f"Parallel found: {len(parallel_matches)} matches")


def test_scalability(workers_list=None):
    """Test how performance scales with number of workers"""
    if workers_list is None:
        cpu_count = mp.cpu_count()
        workers_list = [1, 2, 4, min(8, cpu_count), cpu_count, cpu_count * 2]
        workers_list = sorted(set(workers_list))

    print("\nScalability test: Performance vs Worker Count\n")

    known_image = "dataset/known_woman.jpg"
    image_folder = "dataset/imageset/"

    results = []

    for num_workers in workers_list:
        print(f"Testing with {num_workers} worker(s)...")

        recognizer = ParallelFaceRecognize(known_image, num_workers=num_workers)
        recognizer.load_known_face()

        start = time.time()
        matches = recognizer.search_parallel(image_folder)
        elapsed = time.time() - start

        results.append({
            'workers': num_workers,
            'time': elapsed,
            'matches': len(matches)
        })

        print(f"Time: {elapsed:.2f}s | Matches: {len(matches)}\n")

    # Summary table
    print("Scalability summary")
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12}")
    print("-" * 46)

    baseline_time = results[0]['time']

    for r in results:
        speedup = baseline_time / r['time'] if r['time'] > 0 else 0
        efficiency = (speedup / r['workers']) * 100 if r['workers'] > 0 else 0
        print(f"{r['workers']:<10} {r['time']:<12.2f} {speedup:<12.2f} {efficiency:<12.1f}%")


def test_find_all_vs_first():
    """Compare find-all vs find-first performance"""
    print("\nComparison: Find All vs Find First Match\n")

    known_image = "dataset/known_woman.jpg"
    image_folder = "dataset/imageset/"

    # Find all
    print("Finding ALL matches")
    recognizer_all = ParallelFaceRecognize(known_image)
    recognizer_all.load_known_face()

    start = time.time()
    matches_all = recognizer_all.search_parallel(image_folder, find_all=True)
    time_all = time.time() - start

    print(f"Found {len(matches_all)} matches in {time_all:.2f}s\n")

    # Find first
    print("Finding first match only")
    recognizer_first = ParallelFaceRecognize(known_image)
    recognizer_first.load_known_face()

    start = time.time()
    matches_first = recognizer_first.search_parallel(image_folder, find_all=False)
    time_first = time.time() - start

    print(f"Found {len(matches_first)} match(es) in {time_first:.2f}s\n")

    # Analysis
    print("Analysis:")
    if time_first < time_all:
        speedup = time_all / time_first
        saved = (1 - time_first / time_all) * 100
        print(f"Early termination saved {saved:.1f}% time")
        print(f"Find first: {time_first:.2f}s")
        print(f"Find all:   {time_all:.2f}s")
        print(f"Speedup:    {speedup:.2f}x")
    else:
        print("No significant difference (match found very early)")


if __name__ == "__main__":
    compare_performance(find_all=True)
    test_scalability()
    test_find_all_vs_first()