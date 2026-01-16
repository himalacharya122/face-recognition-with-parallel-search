'''
To install face_recognition, simply use 'pip install face_recognition' in a terminal
However, often you may meet an error about the 'dlib' library with cmake.
The easy solution is to visit https://github.com/z-mahmud22/Dlib_Windows_Python3.x and download the 
compiled wheels locally with the python version, and install it from local

if you want to show the found image with the known face, you need opencv and also uncomment the related code.

'''


import cv2 # type: ignore
import time
import face_recognition # type: ignore
import os

def show_found_image(unknown_image, filename, output_dir="found_by_serial"):
    """
    Refactored from original: Now it save the found image instead of using cv2.imshow to display
    I have done this to ensure it works in my wsl headless environment.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    face_locations = face_recognition.face_locations(unknown_image)

    # convert RGB (face_recognition) to BGR (OpenCV)
    output_image = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

    # draw rectangles around faces
    for top, right, bottom, left in face_locations:
        cv2.rectangle(output_image, (left, top), (right, bottom), (0, 255, 0), 2)

    # logic update: Save the image
    save_path = os.path.join(output_dir, f"found_{filename}")
    cv2.imwrite(save_path, output_image)


def serial_face_recognition(known_image_path, folder_path, find_all=True, save_results=False):
    """
    Provided logic wrapped in a function for benchmarking.
    """
    start = time.time()
    
    # Load the known face image and get the features of the face
    known_image = face_recognition.load_image_file(known_image_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]

    # Get all image files
    filenames = [file.name for file in os.scandir(folder_path) if file.is_file()]
    
    matches = []

    for filename in filenames:
        # Load the unknown face image
        unknown_image = face_recognition.load_image_file(os.path.join(folder_path, filename))

        # Find faces and encodings in the unknown image
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        for unknown_encoding in unknown_encodings:
            # Compare the unknown face encoding with the known encoding
            result = face_recognition.compare_faces([known_encoding], unknown_encoding)

            if result[0]:  # if a match is found
                print("Match found! in " + filename)
                matches.append(filename)
                
                if save_results:
                    show_found_image(unknown_image, filename)
                
                # break out of the face loop if one is found in this image
                break
        
        # benchmark logic: early exit if find_all is False
        if not find_all and matches:
            break

    time_taken = time.time() - start
    return matches, time_taken

if __name__ == "__main__":
    known_path = "dataset/known_woman.jpg"
    images_path = "dataset/imageset/"
    
    found_matches, elapsed = serial_face_recognition(known_path, images_path, find_all=True, save_results=True)
    print(f"Finished in: {elapsed:.2f}s")