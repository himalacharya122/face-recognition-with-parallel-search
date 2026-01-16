"""
Utility functions for parallel face recognition
"""

import os
import pickle
import hashlib
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


def calculate_optimal_workers(total_files, max_workers=None, min_files_per_worker=12):
    """
    Calculate optimal number of workers based on workload
    
    Args:
        total_files: total number of files to process
        max_workers: maximum workers to use (None = auto)
        min_files_per_worker: minimum files each worker should handle
    
    Returns:
        suitable worker count

    Explanation:
        Reserve one core for main process to avoid oversubscription
        Oversubscription = creating more threads than physical cores
    """
    available_cores = get_cpu_count()
    usable_cores = max(1, available_cores - 1)  # reserve 1 for main process
    
    if max_workers is not None:
        usable_cores = min(usable_cores, max_workers)
    
    # calculate based on workload
    # ensure each worker has meaningful work to do
    optimal_for_workload = max(1, total_files // min_files_per_worker)
    
    # choose the smaller - either CPU limit or workload-appropriate
    optimal = min(usable_cores, optimal_for_workload)
    
    # for very small datasets, use fewer workers
    if total_files <= 5:
        optimal = min(2, optimal)
    
    return max(1, optimal)


def chunk_list(lst, n):
    """
    Split list into n roughly equal chunks
    
    Args:
        lst: list to split
        n: number of chunks
    
    Returns:
        List of chunks
        
    Explanation:
        Load-balanced chunking - distributes items as evenly as possible
        Example: 10 items, 3 workers - [4, 3, 3] instead of [3, 3, 4]
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


def get_file_hash(filepath):
    """
    Calculate MD5 hash of file for cache key
    
    Args:
        filepath: path to file
    
    Returns:
        MD5 hash string
    
    Explanation:
        Used to detect if image has changed since last cache
        Only first 8KB is hashed for speed (sufficient for detection)
    """
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            # read first 8KB for speed
            chunk = f.read(8192)
            hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None


class FaceCache:
    """
    Cache system for face encodings and locations
    
    Time Complexity: O(1) for cache hits, O(n) for cache misses
    Space Complexity: O(n) where n = number of cached images
    
    Explanation:
        Stores face locations and encodings to disk to avoid recomputation
        Uses pickle for serialization and MD5 hashes for cache validation
    """
    
    def __init__(self, cache_dir=".face_cache"):
        """
        Initialize cache
        
        Args:
            cache_dir: directory to store cache files
        """
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_path(self, image_path):
        """Get cache file path for given image"""
        filename = os.path.basename(image_path)
        cache_filename = f"{filename}.cache"
        return os.path.join(self.cache_dir, cache_filename)
    
    def get(self, image_path):
        """
        Get cached face data
        
        Args:
            image_path: path to image file
        
        Returns:
            dict with 'locations' and 'encodings', or None if not cached/invalid
        
        Explanation:
            Validates cache using file hash to ensure image hasn't changed
        """
        cache_path = self._get_cache_path(image_path)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # validate cache using file hash
            current_hash = get_file_hash(image_path)
            if cached_data.get('hash') != current_hash:
                # image changed, cache invalid
                return None
            
            return {
                'locations': cached_data['locations'],
                'encodings': cached_data['encodings']
            }
            
        except Exception:
            return None
    
    def set(self, image_path, locations, encodings):
        """
        Store face data in cache
        
        Args:
            image_path: path to image file
            locations: face locations from face_recognition
            encodings: face encodings from face_recognition
        
        Explanation:
            Stores both data and file hash for validation
        """
        cache_path = self._get_cache_path(image_path)
        
        try:
            cache_data = {
                'hash': get_file_hash(image_path),
                'locations': locations,
                'encodings': encodings
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            print(f"Warning: Failed to cache {image_path}: {e}")
    
    def clear(self):
        """Clear all cache files"""
        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            print(f"Cache cleared: {self.cache_dir}")
        except Exception as e:
            print(f"Error clearing cache: {e}")