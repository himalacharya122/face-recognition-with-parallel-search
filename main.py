"""
Main entry point for parallel face recognition with caching
"""
import time
import os
import cv2 # type: ignore
from parallel_face_search import ParallelFaceRecognize
import multiprocessing as mp
import face_recognition # type: ignore
from tqdm import tqdm
from utils import FaceCache

def save_found_images(matches, folder_path, known_encoding, output_dir="found_by_parallel", use_cache=True):
    """
    Save found image with rectangle (bounding box) around detected face
    Rectangle will apply only to the target face
    
    Args:
        matches: list of matched filenames
        folder_path: source folder path
        known_encoding: encoded known face
        output_dir: directory to save results
        use_cache: whether to use caching (default True)
    
    Explanation:
        Cache stores face locations and encodings to avoid recomputation
        Speeds up repeated runs significantly
    """
    # create output directory if doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # initialize cache
    cache = FaceCache() if use_cache else None
    
    cache_hits = 0
    cache_misses = 0
    
    print(f"Processing {len(matches)} matched images...")
    for filename in matches:
        try:
            image_path = os.path.join(folder_path, filename)
            
            # try to get from cache first
            cached_data = cache.get(image_path) if cache else None
            
            if cached_data:
                # cache hit - use cached data
                face_locations = cached_data['locations']
                face_encodings = cached_data['encodings']
                cache_hits += 1
                
                # still need to load image for drawing
                image = face_recognition.load_image_file(image_path)
            else:
                # cache miss - compute and cache
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                # store in cache for next time
                if cache:
                    cache.set(image_path, face_locations, face_encodings)
                cache_misses += 1
            
            # convert RGB (face_recognition) to BGR (OpenCV)
            output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # loop through each face found in the image
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # check if this specific face matches the known target
                matches_target = face_recognition.compare_faces([known_encoding], face_encoding)
                
                # only draw if this specific face is the one we want
                if matches_target[0]: # if it's a match
                    cv2.rectangle(output_image, (left, top), (right, bottom), (0, 255, 0), 2)
                    # add a label
                    cv2.putText(output_image, "MATCH", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            save_path = os.path.join(output_dir, f"found_{filename}")
            cv2.imwrite(save_path, output_image)
            
        except Exception as e:
            print(f"  Error saving {filename}: {e}")
    
    # print cache statistics
    if use_cache:
        total = cache_hits + cache_misses
        hit_rate = (cache_hits / total * 100) if total > 0 else 0
        print(f"\nCache Statistics:")
        print(f"Cache hits: {cache_hits}/{total} ({hit_rate:.1f}%)")
        print(f"Cache misses: {cache_misses}/{total}")
        if cache_hits > 0:
            print(f"Caching saved a lot of computation time!")


def main():
    """
    Main execution function
    """
    print("-------------------------------------------")
    print("Parallel face recognition")
    print("-------------------------------------------\n")
    
    # image paths
    known_image = "dataset/max.jpg"
    image_folder = "dataset/imageset/"
    
    # initialize parallel recognizer
    recognizer = ParallelFaceRecognize(known_image)
    recognizer.load_known_face()
    
    # get image files
    print("\nScanning image folder...")
    filenames = recognizer.get_image_files(image_folder)
    print(f"Found {len(filenames)} images to process")

    # calculate no. of workers
    if recognizer.num_workers is None:
        from utils import calculate_optimal_workers
        recognizer.num_workers = calculate_optimal_workers(len(filenames))
    print(f"Using {recognizer.num_workers} worker processes")
    print(f"System detected {mp.cpu_count()} logical CPU threads")
    
    # parallel search
    print("\nSearching for matching faces (parallel mode)...")
    start_time = time.time()
    
    matches = recognizer.search_parallel(image_folder)
    
    parallel_time = time.time() - start_time
    
    # display results
    print("\nProcessing results...")
    
    if matches:
        print(f"\nMatches found in {len(matches)} image(s):")
        for i, filename in enumerate(matches, 1):
            print(f"{i}. {filename}")
        
        # save found images with rectangles with caching
        print(f"\nSaving results to 'found_by_parallel/' directory...")
        save_start = time.time()
        save_found_images(matches, image_folder, recognizer.known_encoding, use_cache=True)
        save_time = time.time() - save_start
        
        print(f"\nImages saved in {save_time:.2f} seconds")
        
    else:
        print("\nNo matches found")
    
    # performance metrics
    print("\nPerformance metrics:")
    print(f"Search time: {parallel_time:.2f} seconds")
    print(f"Workers used: {recognizer.num_workers}")
    print(f"Images per worker: ~{len(filenames) // recognizer.num_workers}")
    print(f"Average time per image: {parallel_time / len(filenames):.3f} seconds")
    
    print("\nTip: Run again to see cache speedup!")
    print("Run 'python benchmark.py' to compare with serial performance")

if __name__ == "__main__":
    main()