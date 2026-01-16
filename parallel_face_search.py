"""
Parallel Face Recognition using multiprocessing
Uses utility functions
"""

import time
import multiprocessing as mp
import face_recognition # type: ignore
import os
from tqdm import tqdm
from functools import partial
from utils import calculate_optimal_workers, chunk_list, validate_image_path

class ParallelFaceRecognize:
    """
    Parallel face recognition using process pool
    
    Time Complexity: O(n/p) where n = images, p = processes
    Space Complexity: O(n) for image data

    Key Concept: Data Decomposition (from Flynn's Taxonomy - MIMD)
    Each worker processes a different subset of images independently
    """
    
    def __init__(self, known_image_path, num_workers=None):
        """
        Initialize recognizer
        
        Args:
            known_image_path: path to known face image
            num_workers: number of worker processes (None = auto-detect)
        """
        if not validate_image_path(known_image_path):
            raise ValueError(f"Invalid known image path: {known_image_path}")

        self.known_image_path = known_image_path
        self.known_encoding = None
        self.num_workers = num_workers
    
    def load_known_face(self):
        """
        Load and encode the known face
        
        Time: O(1) - done once

        Explanation:
            Done only once in the main process, then shared (read-only) with all workers to avoid memory duplication
        """
        print(f"Loading known face from: {self.known_image_path}")
        known_image = face_recognition.load_image_file(self.known_image_path)
        encodings = face_recognition.face_encodings(known_image)

        if not encodings:
            raise ValueError("No face found in known image!")
        
        self.known_encoding = encodings[0]
        print("Known face loaded successfully")
        
    
    def get_image_files(self, folder_path):
        """
        Get list of all image files in folder
        
        Returns:
            List of image filenames
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        filenames = []
        for file in os.scandir(folder_path):
            if file.is_file():
                _, ext = os.path.splitext(file.name.lower())
                if ext in valid_extensions:
                    filenames.append(file.name)
        
        return filenames
    
    def search_parallel(self, folder_path, find_all=True):
        """
        Search for matching faces in parallel

        Args:
            folder_path: directory containing images
            find_all: if True, find all matches; if False, stop at first match

        Returns:
            List of filenames with matching faces

        Explanation:
            Heart of parallel processing
            1. create chunks aka data decomposition
            2. spawn worker pool
            3. map chunks to workers
            4. collect results

            Uses multiprocessing.Pool which implements MIMD (Multiple Instruction Multiple Data) from Flynn's Taxonomy
        """

        # get all image files
        filenames = self.get_image_files(folder_path)
        
        if not filenames:
            print("No image files found!")
            return []
        
        # calculate optimal workers based on workload
        if self.num_workers is None:
            self.num_workers = calculate_optimal_workers(
                total_files=len(filenames)
            )
        
        # create chunks for workers
        chunks = chunk_list(filenames, self.num_workers)

        print(f"\nProcessing {len(filenames)} images with {self.num_workers} workers")
        print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")

        # create process pool and process chunks in parallel
        with mp.Pool(processes=self.num_workers) as pool:
            # use partial to fix folder_path argument
            process_func = partial(
                self._process_chunk, 
                folder_path=folder_path,
                find_all=find_all
            )

            # map chunks to workers (parallel execution)
            # use imap to track progress by chunks
            results = []
            with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
                for result in pool.imap(process_func, chunks):
                    results.append(result)
                    pbar.update(1)

        # flatten results from all workers
        matches = []
        for worker_matches in results:
            matches.extend(worker_matches)

        return matches
    
    def _process_chunk(self, chunk, folder_path, find_all=True):
        """
        Process a chunk of images (runs in worker process)

        Args:
            chunk: list of filenames to process
            folder_path: directory containing images
            find_all: if True, process all images; if False, stop at first match
        
        Returns:
            List of matched filenames

        Explanation:
            This runs in parallel in separate processes
            Each process has its own copy of Python Interpreter to avoid GIL
        """
        matches = []

        for filename in chunk:
            result = self._process_single_image(filename, folder_path)
            if result:
                matches.append(filename)

                # early exit if only need the first match
                if not find_all:
                    break

        return matches
    
    def _process_single_image(self, filename, folder_path):
        """
        Process single image for face matching
        
        Time: O(1) per image
        Space: O(1)
        
        Returns:
            filename if match found, None otherwise

        Explanation:
            Early exit optimization: returns immediately on first match.
            No need to check remaining faces in same image.
        """
        try:
            image_path = os.path.join(folder_path, filename)
            unknown_image = face_recognition.load_image_file(image_path)
            
            # get all face encodings in this image
            unknown_encodings = face_recognition.face_encodings(unknown_image)
            
            # no faces found
            if not unknown_encodings:
                return None
            
            # check each face (early exit on first match)
            for unknown_encoding in unknown_encodings:
                matches = face_recognition.compare_faces(
                    [self.known_encoding], 
                    unknown_encoding
                )
                
                if matches[0]:  # match found!
                    return filename
            
            return None
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None


def process_image_worker(filename, known_encoding, folder_path):
    """
    Worker function for processing single image
    Must be at module level for multiprocessing.Pool
    
    Args:
        filename: image filename
        known_encoding: encoded known face
        folder_path: directory path
    
    Returns:
        filename if match found, None otherwise
    """
    try:
        image_path = os.path.join(folder_path, filename)
        unknown_image = face_recognition.load_image_file(image_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        
        if not unknown_encodings:
            return None
        
        for unknown_encoding in unknown_encodings:
            matches = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if matches[0]:
                return filename
        
        return None
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None