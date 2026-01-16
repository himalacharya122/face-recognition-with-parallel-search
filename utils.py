"""
Utility functions for parallel face recognition
"""

import os
import multiprocessing as mp


def get_cpu_count():
    """
    Get number of available CPU cores
    
    Returns:
        Number of CPU cores

    Explanation:
        Use multiprocessing's cpu count to detect hardware threads
        Important for determining parallel capacity
    """
    return mp.cpu_count()


def calculate_optimal_workers(total_files, max_workers=None):
    """
    Calculate optimal number of workers based on workload
    
    Args:
        total_files: total number of files to process
        max_workers: maximum workers to use
    
    Returns:
        suitable worker count

    Explanation:
        Reserve one core for main process to avoid oversubscription
        Oversubscription = creating more threads than physical cores
    """
    available_cores = get_cpu_count()
    usable_cores = max(1, available_cores - 1)
    
    if max_workers is not None:
        usable_cores = min(usable_cores, max_workers)
    
    # ensure at least 15 images per worker for efficiency
    # this prevents over-parallelization
    optimal_for_workload = max(1, total_files // 15)
    
    # choose the smaller - either CPU limit or workload-appropriate
    optimal = min(usable_cores, optimal_for_workload)
    
    return max(1, optimal)


def chunk_list(lst, n):
    """
    Split list into n roughly equal chunks
    
    Args:
        lst: list to split
        n: number of chunks
    
    Returns:
        List of chunks
    """
    if n <= 0:
        return []
    if n >= len(lst):
        # more workers than items, return one item per chunk
        return [[item] for item in lst]
    
    chunk_size = len(lst) // n
    remainder = len(lst) % n

    chunks = []
    start = 0

    for i in range(n):
        # first 'remainder' chunks get an extra item
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    
    return chunks


def validate_image_path(path):
    """
    Validate if path is valid image file
    
    Returns:
        bool: True if valid
    """
    if not os.path.exists(path):
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    _, ext = os.path.splitext(path.lower())

    return ext in valid_extensions

def format_time(seconds):
    """
    Format seconds into human-readable time string
    
    Args:
        seconds: time in seconds
    
    Returns:
        Formatted string (e.g., "2m 15s" or "45.2s")
    """
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{seconds:.2f}s"